"""Model loading, tokenizer training, collator, and augmentation utilities."""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .config import (
    LEARNING_RATE_DECODER,
    LEARNING_RATE_ENCODER,
)
from .text import collect_text_corpora

logger = logging.getLogger(__name__)

_LANGUAGE_TOKENS = config.CONFIG["model"]["language_tokens"]
_BPE_SYMBOLS = [
    *config.WOLOF_SPECIAL_CHARS,
    "î",
    "ô",
    "û",
    *_LANGUAGE_TOKENS,
]


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, numpy, and torch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class DataCollatorSpeechSeq2SeqWithPadding:
    """Build a Whisper batch without materializing features in RAM.

    Accepts raw rows (``audio`` path + ``transcript``) and extracts Mel features
    lazily per batch, or precomputed rows (``input_features`` + tokenized labels),
    e.g. produced by NoisyDataset.
    """

    def __init__(self, processor: Any, padding: str = "longest"):
        self.processor = processor
        self.padding = padding

    @staticmethod
    def _mel(processor: Any, audio: np.ndarray) -> np.ndarray:
        """Extract the 2D Mel feature array for one clip."""
        return processor.feature_extractor(
            audio,
            sampling_rate=config.TARGET_SAMPLE_RATE,
            return_tensors="pt",
        ).input_features[0]

    def _raw_features(self, features: list[dict]) -> list[dict]:
        """Load each raw audio clip and extract its Mel features."""
        import librosa

        batches: list[dict] = []
        for f in features:
            audio = f["audio"]
            if isinstance(audio, dict):
                array = audio["array"]
            else:
                array, _ = librosa.load(audio, sr=config.TARGET_SAMPLE_RATE)
            batches.append({"input_features": self._mel(self.processor, array)})
        return batches

    def __call__(self, features: list[dict]) -> dict:
        precomputed = "input_features" in features[0]
        if precomputed:
            raw_batch = [{"input_features": f["input_features"]} for f in features]
            label_ids = [f["labels"] for f in features]
        else:
            raw_batch = self._raw_features(features)
            label_ids = [
                self.processor.tokenizer(f["transcript"], add_special_tokens=False).input_ids
                for f in features
            ]

        batch = self.processor.feature_extractor.pad(
            raw_batch,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            {"input_ids": label_ids},
            padding=self.padding,
            return_tensors="pt",
        )["input_ids"]
        max_len = self.processor.tokenizer.model_max_length
        if labels.shape[1] > max_len:
            labels = labels[:, :max_len]
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class NoisyDataset:
    """Dataset wrapper applying randomized audio augmentation."""

    def __init__(
        self, dataset: Any, processor: Any, noise_dir: Path | None = None, aug_prob: float = 0.5
    ):
        self.dataset = dataset
        self.processor = processor
        self.noise_dir = noise_dir
        self.aug_prob = aug_prob

    def __len__(self) -> int:
        return len(self.dataset)

    def _augment(self, array: np.ndarray, sr: int = config.TARGET_SAMPLE_RATE) -> np.ndarray:
        """Apply time-stretch, pitch-shift, and noise augmentation."""
        import librosa

        if random.random() < 0.3:
            rate = random.uniform(0.9, 1.1)
            array = librosa.effects.time_stretch(array, rate=rate)
        if random.random() < 0.3:
            n_steps = random.uniform(-2, 2)
            array = librosa.effects.pitch_shift(array, sr=sr, n_steps=n_steps)
        if random.random() < 0.3:
            noise = np.random.randn(len(array)) * 0.005 * random.uniform(0.5, 1.5)
            array = array + noise
        if self.noise_dir and random.random() < 0.2:
            noise_files = list(self.noise_dir.glob("*.wav"))
            if noise_files:
                noise_audio, _ = librosa.load(str(random.choice(noise_files)), sr=sr)
                if len(noise_audio) > len(array):
                    noise_audio = noise_audio[: len(array)]
                else:
                    reps = len(array) // len(noise_audio) + 1
                    noise_audio = np.tile(noise_audio, reps)[: len(array)]
                scale = random.uniform(0.01, 0.1)
                array = array + noise_audio * scale / max(np.max(np.abs(noise_audio)), 1e-8)
        peak = np.max(np.abs(array))
        if peak > 0:
            array = array / peak * 0.99
        return array

    def __getitem__(self, idx: int) -> dict:
        ex = self.dataset[idx]
        audio = ex["audio"]["array"] if isinstance(ex["audio"], dict) else ex["audio"]
        if random.random() < self.aug_prob:
            audio = self._augment(audio)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features[0]
        labels = self.processor.tokenizer(
            ex["transcript"],
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        return {"input_features": input_features, "labels": labels}


def train_bpe(corpus: str, vocab_size: int = config.DATA["tokenizer"]["vocab_size"]) -> dict:
    """Train a Wolof-optimized BPE vocabulary with SentencePiece.

    Returns a dict with the actual `É`, `É`, `Ë` and `<pad>` ids read
    from the trained model (never hardcoded), plus the remaining pieces.
    """
    import tempfile

    import sentencepiece as spm

    # Use temp directory to avoid Windows path issues
    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_file = Path(tmp_dir) / "corpus.txt"
        corpus_file.write_text(corpus, encoding="utf-8")
        model_prefix = str(Path(tmp_dir) / "wolof")

        symbols = [*_BPE_SYMBOLS, "<pad>"]
        spm.SentencePieceTrainer.train(
            input=corpus_file.as_posix(),
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=1.0,
            byte_fallback=True,
            num_threads=8,
            user_defined_symbols=symbols,
            normalization_rule_name="identity",
        )
        model = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

        special_ids = {
            "unk": model.piece_to_id("É"),
            "bos": model.piece_to_id("É"),
            "eos": model.piece_to_id("Ë"),
            "pad": model.piece_to_id("<pad>"),
        }
        pieces = [
            model.id_to_piece(i)
            for i in range(model.get_piece_size())
            if i not in special_ids.values()
        ]
        return {**special_ids, "pieces": pieces}


def extend_whisper_tokenizer(new_pieces: list[str], model_path: str) -> str:
    """Extend a Whisper tokenizer with Wolof pieces and language tokens.

    The extended tokenizer is saved to ``config.EXTENDED_TOKENIZER_DIR`` so that
    ``build_model`` can load it and resize the model embeddings accordingly.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)

    for piece in new_pieces:
        if piece not in tokenizer.get_vocab():
            tokenizer.add_tokens(piece)
    for lang_token in _LANGUAGE_TOKENS:
        if lang_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [lang_token]})

    out_dir = Path(config.EXTENDED_TOKENIZER_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Tokenizer saved to %s (vocab size %d)", out_dir, tokenizer.vocab_size)
    return str(out_dir)


def train_tokenizer_pipeline(
    vocab_size: int = config.DATA["tokenizer"]["vocab_size"],
    extend_whisper: bool = False,
    model_path: str = config.BASE_MODEL,
) -> dict:
    """Train the Wolof BPE tokenizer and optionally extend the Whisper tokenizer."""
    texts = collect_text_corpora(config.DATA["tokenizer"]["corpora_paths"])
    if not texts:
        raise FileNotFoundError("No Wolof text corpora found. Place text files in data/text/wolof/")
    corpus = "\n".join(texts)
    vocab = train_bpe(corpus, vocab_size=vocab_size)

    saved_tokenizer: str | None = None
    if extend_whisper:
        saved_tokenizer = extend_whisper_tokenizer(vocab["pieces"], model_path)

    result: dict = {
        "n_pieces": len(vocab["pieces"]),
        "corpus_chars": len(corpus),
        "extended_tokenizer": saved_tokenizer,
    }
    return result


def build_model(
    base_model: str,
    use_lora: bool = False,
    resume_from: str | None = None,
    lr_decoder: float = LEARNING_RATE_DECODER,
    lr_encoder: float = LEARNING_RATE_ENCODER,
    tokenizer_path: str | None = None,
) -> tuple[Any, list[dict]]:
    """Load a Whisper model, optionally wrap it in LoRA, and build discrimative LR groups.

    When *tokenizer_path* points to an extended Whisper tokenizer (produced by
    ``extend_whisper_tokenizer``), the model embedding layer is resized so that
    all newly added tokens have an embedding.
    """
    from transformers import WhisperForConditionalGeneration

    if resume_from:
        logger.info("Resuming from %s", resume_from)
        model = WhisperForConditionalGeneration.from_pretrained(resume_from)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(base_model)

    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(
            "Resized embeddings to %d tokens (extended tokenizer)",
            len(tokenizer),
        )

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.config.language = config.LANGUAGE
    model.config.task = "transcribe"

    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    param_groups = [
        {"params": [p for p in decoder_params if p.requires_grad], "lr": lr_decoder},
        {"params": [p for p in encoder_params if p.requires_grad], "lr": lr_encoder},
    ]
    return model, param_groups
