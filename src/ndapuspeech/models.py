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


WHISPER_EXPECTED_MEL_FRAMES: int = 3000

_NO_TIMESTAMPS_TOKEN = "<|notimestamps|>"
_EOT_TOKEN = "<|endoftext|>"


def build_decoder_labels(tokenizer: Any, text: str) -> list[int]:
    """Token ids for free-transcription Whisper labels.

    Matches Whisper's free-transcription decoder prompt
    ``[<|startoftranscript|>, <|notimestamps|>]``: labels start right after the
    decoder start token with ``<|notimestamps|>`` and end with ``<|endoftext|>``.
    No language or task token is prepended, so the fine-tuned model never
    depends on a language the base vocabulary does not contain.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    no_ts = tokenizer.convert_tokens_to_ids(_NO_TIMESTAMPS_TOKEN)
    eot = tokenizer.convert_tokens_to_ids(_EOT_TOKEN)
    return [no_ts, *ids, eot]


def configure_free_transcription(model: Any) -> None:
    """Disable Whisper's language detection / forced prefix for fine-tuning.

    The model is fine-tuned without a language token (Wolof has none in the base
    vocabulary). Generation must therefore start from the bare prompt
    ``[<|startoftranscript|>, <|notimestamps|>]`` instead of running language
    detection and injecting a wrong language token. This mirrors the labels
    produced by :func:`build_decoder_labels`.
    """
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.config.language = None
    model.config.task = None
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        for attr in ("forced_decoder_ids", "lang_to_id", "task_to_id"):
            gen_cfg.__dict__.pop(attr, None)
        gen_cfg.language = None
        gen_cfg.task = None
        gen_cfg.suppress_tokens = None
        gen_cfg.begin_suppress_tokens = None


class DataCollatorSpeechSeq2SeqWithPadding:
    """Build a Whisper batch without materializing features in RAM.

    Accepts raw rows (``audio`` path + ``transcript``) and extracts Mel features
    lazily per batch, or precomputed rows (``input_features`` + tokenized labels),
    e.g. produced by NoisyDataset.

    Whisper's encoder requires exactly 3000 mel frames (30s). Precomputed
    features are stored trimmed to save mmap space, so every batch is
    re-padded/truncated to ``target_frames`` here.
    """

    def __init__(
        self,
        processor: Any,
        padding: str = "longest",
        target_frames: int | None = None,
    ):
        self.processor = processor
        self.padding = padding
        if target_frames is None:
            target_frames = int(
                getattr(processor.feature_extractor, "nb_max_frames", 0)
                or WHISPER_EXPECTED_MEL_FRAMES
            )
        self.target_frames: int = target_frames

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
        import soundfile as sf
        from librosa import resample

        batches: list[dict] = []
        for f in features:
            audio = f["audio"]
            if isinstance(audio, dict):
                array = audio["array"]
                audio_sr = audio.get("sampling_rate")
                if audio_sr and audio_sr != config.TARGET_SAMPLE_RATE:
                    array = resample(
                        array,
                        orig_sr=audio_sr,
                        target_sr=config.TARGET_SAMPLE_RATE,
                    )
            else:
                array, sr = sf.read(audio, dtype="float32")
                if sr != config.TARGET_SAMPLE_RATE:
                    array = resample(array, orig_sr=sr, target_sr=config.TARGET_SAMPLE_RATE)
            batches.append({"input_features": self._mel(self.processor, array)})
        return batches

    def _pad_mels(self, mels: list[np.ndarray]) -> list[dict]:
        """Pad/trim (n_mels, time) arrays to exactly ``self.target_frames``.

        Whisper's encoder requires 3000 frames; ``feature_extractor.pad``
        with ``padding="longest"`` keeps shorter batches short, hence the
        explicit rectangularization here.
        """
        padded: list[dict] = []
        for m in mels:
            if hasattr(m, "detach"):
                m = m.detach().cpu().numpy()  # type: ignore[union-attr]
            arr = np.asarray(m, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D mel (n_mels, time), got shape {arr.shape}")
            n_frames = arr.shape[1]
            if n_frames > self.target_frames:
                arr = arr[:, : self.target_frames]
            elif n_frames < self.target_frames:
                arr = np.pad(arr, ((0, 0), (0, self.target_frames - n_frames)))
            padded.append({"input_features": arr})
        return padded

    def __call__(self, features: list[dict]) -> dict:
        precomputed = "input_features" in features[0]
        if precomputed:
            raw_batch = self._pad_mels(
                [np.asarray(f["input_features"], dtype=np.float32) for f in features]
            )
            label_ids = [f["labels"] for f in features]
        else:
            raw_mels = [
                np.asarray(d["input_features"], dtype=np.float32)
                for d in self._raw_features(features)
            ]
            raw_batch = self._pad_mels(raw_mels)
            label_ids = [
                build_decoder_labels(self.processor.tokenizer, f["transcript"]) for f in features
            ]

        batch = self.processor.feature_extractor.pad(
            raw_batch,
            padding=self.padding,
            return_tensors="pt",
        )
        import torch

        if batch["input_features"].dtype == torch.float64:
            batch["input_features"] = batch["input_features"].to(torch.float32)
        max_len = self.processor.tokenizer.model_max_length
        padded = [list(ids)[:max_len] for ids in label_ids]
        width = max(len(ids) for ids in padded)
        labels = torch.tensor(
            [ids + [-100] * (width - len(ids)) for ids in padded], dtype=torch.long
        )
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
        raw_audio = ex["audio"]["array"] if isinstance(ex["audio"], dict) else ex["audio"]
        if isinstance(raw_audio, str):
            import soundfile as sf
            from librosa import resample

            array, sr = sf.read(raw_audio, dtype="float32")
            if sr != config.TARGET_SAMPLE_RATE:
                array = resample(
                    array, orig_sr=sr, target_sr=config.TARGET_SAMPLE_RATE
                ).astype(np.float32)
            else:
                array = np.asarray(array, dtype=np.float32)
        else:
            array = np.asarray(raw_audio, dtype=np.float32)
        if random.random() < self.aug_prob:
            array = self._augment(array)
        input_features = self.processor.feature_extractor(
            array, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features[0]
        labels = build_decoder_labels(self.processor.tokenizer, ex["transcript"])
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
    attn_implementation: str | None = None,
    load_in_4bit: bool = False,
    torch_compile: bool = False,
) -> tuple[Any, list[dict]]:
    """Load a Whisper model, optionally wrap it in LoRA, and build discrimative LR groups.

    When *tokenizer_path* points to an extended Whisper tokenizer (produced by
    ``extend_whisper_tokenizer``), the model embedding layer is resized so that
    all newly added tokens have an embedding.

    *attn_implementation* forces the CUDA kernel (``"sdpa"`` on RTX 5060).
    *load_in_4bit* enables QLoRA via bitsandbytes to fit 8 Go VRAM.
    *torch_compile* fuses kernels with ``torch.compile`` (2-3 min warmup).
    """
    from transformers import WhisperForConditionalGeneration

    load_kwargs: dict[str, Any] = {}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        try:
            import torch
            from transformers import BitsAndBytesConfig

            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["device_map"] = "auto"
        except (ImportError, RuntimeError) as e:
            logger.warning("4-bit quantization unavailable (%s); using fp16 weights", e)

    if resume_from:
        logger.info("Resuming from %s", resume_from)
        model = WhisperForConditionalGeneration.from_pretrained(resume_from, **load_kwargs)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(base_model, **load_kwargs)

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
        if load_in_4bit and "quantization_config" in load_kwargs:
            try:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)
            except (ImportError, ValueError) as e:
                logger.warning("k-bit training prep skipped (%s)", e)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if torch_compile:
        try:
            import torch

            model = torch.compile(model)
            logger.info("torch.compile enabled (fused kernels)")
        except RuntimeError as e:
            logger.warning("torch.compile failed (%s); continuing uncompiled", e)

    configure_free_transcription(model)

    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    param_groups = [
        {"params": [p for p in decoder_params if p.requires_grad], "lr": lr_decoder},
        {"params": [p for p in encoder_params if p.requires_grad], "lr": lr_encoder},
    ]
    return model, param_groups


def merge_lora_weights(model: Any) -> Any:
    """Merge LoRA adapters into the base model, returning a standalone model.

    Returns *model* unchanged when it carries no LoRA layers or the merge fails
    (e.g. quantized weights that peft cannot merge on this host).
    """
    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        return model
    try:
        return merge()
    except (ValueError, RuntimeError) as e:
        logger.warning("LoRA merge failed (%s); saving adapter only", e)
        return model


def merge_saved_lora_adapter(model_dir: Path | str, base_model: str | None = None) -> Path:
    """Merge a saved LoRA adapter into its base model as a standalone model.

    Loads the base model on CPU, pre-resizes its embeddings to the saved
    extended tokenizer (adapters trained here include resized embeddings),
    merges, then writes ``model.safetensors`` next to the tokenizer. Adapter
    files are moved to ``lora_adapter/`` so transformers loads the directory
    directly, without PEFT.
    """
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, WhisperForConditionalGeneration

    out_dir = Path(model_dir)
    adapter_file: Path | None = None
    for candidate in ("adapter_model.safetensors", "adapter_model.bin"):
        if (out_dir / candidate).exists():
            adapter_file = out_dir / candidate
            break
    if adapter_file is None:
        raise FileNotFoundError(f"No LoRA adapter weights in {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
    adapter_cfg = json.loads((out_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_id = base_model or str(adapter_cfg["base_model_name_or_path"])
    base = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float32)
    if len(tokenizer) != base.config.vocab_size:
        base.resize_token_embeddings(len(tokenizer))
        logger.info("Resized base embeddings to %d tokens before merge", len(tokenizer))
    merged = PeftModel.from_pretrained(base, str(out_dir)).merge_and_unload()
    backup = out_dir / "lora_adapter"
    backup.mkdir(exist_ok=True)
    (out_dir / "adapter_config.json").replace(backup / "adapter_config.json")
    adapter_file.replace(backup / adapter_file.name)
    merged.save_pretrained(str(out_dir))
    logger.info("Merged LoRA saved as standalone model in %s", out_dir)
    return out_dir
