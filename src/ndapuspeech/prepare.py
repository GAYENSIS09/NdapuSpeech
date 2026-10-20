"""One-shot Whisper feature precomputation (note.txt §A).

Decodes every JSONL row once into Whisper log-mel ``input_features`` plus
tokenized ``labels`` and persists the result with ``save_to_disk``. Training
then becomes a plain RAM -> VRAM copy with zero signal processing (0x I/O).

Mel is stored trimmed to the true frame count in float16 so the full 80h
dataset stays within the ~8-10 Go mmap budget; batches are re-padded
on the fly by the data collator. Readers use memory-mapping, never full RAM.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .data import resolve_audio_path

logger = logging.getLogger(__name__)

_JSON_SPLITS = ("train", "val", "test")


def is_preprocessed(output_dir: Path) -> bool:
    """Return True when an Arrow DatasetDict was already saved to *output_dir*."""
    return (output_dir / "dataset_dict.json").is_file()


def load_clip(ref: Any) -> tuple[np.ndarray, int]:
    """Decode one audio reference (file path or pre-decoded dict) to mono PCM."""
    if isinstance(ref, dict) and "array" in ref:
        array = np.asarray(ref["array"], dtype=np.float32)
        sr = int(ref.get("sampling_rate") or config.TARGET_SAMPLE_RATE)
        return array, sr
    target = resolve_audio_path(str(ref))
    if target is None:
        raise FileNotFoundError(f"Audio file not found: {ref}")
    import soundfile as sf

    data, sr = sf.read(str(target), dtype="float32", always_2d=False)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=tuple(range(1, array.ndim))).astype(np.float32)
    return array, int(sr)


def _audio_available(example: dict[str, Any]) -> bool:
    """Return True when a dataset row's audio can be resolved to an existing file."""
    ref = example.get("audio")
    if isinstance(ref, dict):
        return True
    return resolve_audio_path(str(ref)) is not None


def _precomputed_features(base: Any) -> Any:
    """Build explicit Arrow features: float16 mel, int64 labels/length.

    Without explicit features, Arrow infers float64 from Python floats and a
    full-size dataset would exceed the ~8-10 Go mmap budget.
    """
    from datasets import Features, Sequence, Value

    fields: dict[str, Any] = dict(base)
    for dropped in ("audio", "transcript"):
        fields.pop(dropped, None)
    fields["input_features"] = Sequence(Sequence(Value("float16")))
    fields["labels"] = Sequence(Value("int64"))
    fields["duration"] = Value("float64")
    fields["length"] = Value("int64")
    return Features(fields)


def encode_example(
    audio_array: np.ndarray,
    sampling_rate: int,
    text: str,
    processor: Any,
) -> dict[str, Any]:
    """Encode one clip into Whisper log-mel features and token ids."""
    from librosa import resample

    array = np.asarray(audio_array, dtype=np.float32)
    if sampling_rate != config.TARGET_SAMPLE_RATE:
        array = resample(
            array,
            orig_sr=sampling_rate,
            target_sr=config.TARGET_SAMPLE_RATE,
        ).astype(np.float32)
    features = processor.feature_extractor(
        array,
        sampling_rate=config.TARGET_SAMPLE_RATE,
        return_tensors="pt",
    ).input_features[0]
    if hasattr(features, "tolist"):
        mel: Any = features.tolist()
    else:
        mel = np.asarray(features).tolist()
    from .models import build_decoder_labels

    label_ids: Any = build_decoder_labels(processor.tokenizer, text)
    return {"input_features": mel, "labels": list(label_ids)}


def prepare_feature_dataset(
    data_dir: Path,
    output_dir: Path,
    processor_id: str,
    tokenizer_dir: str | None = None,
    num_proc: int = 4,
    batch_size: int = 16,
    force: bool = False,
) -> Path:
    """Precompute Mel + labels for every JSONL split and save Arrow to *output_dir*.

    Uses ``datasets.map`` with multiprocessing and ``remove_columns`` so the
    saved table only holds ``input_features``/``labels`` (+ metadata).

    Audio is decoded with soundfile (never the ``datasets`` Audio decoder),
    so no ``torchcodec`` dependency is required. Pass ``force=True`` to rebuild
    an existing cache (e.g. after changing the label/prompt convention).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, WhisperProcessor

    if is_preprocessed(output_dir) and not force:
        logger.info("[SKIP] Preprocessed dataset already at %s", output_dir)
        return output_dir
    data_files = {}
    for s in _JSON_SPLITS:
        candidate = data_dir / f"{s}.jsonl"
        if candidate.exists() and candidate.stat().st_size > 0:
            data_files[s] = str(candidate)
        else:
            logger.warning("[SKIP] Empty or missing split %s in %s", s, data_dir)
    if not data_files:
        raise FileNotFoundError(f"No non-empty JSONL splits in {data_dir}")

    processor = WhisperProcessor.from_pretrained(processor_id)
    if tokenizer_dir:
        processor.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.info("Using extended tokenizer from %s", tokenizer_dir)

    ds = load_dataset("json", data_files=data_files)
    prepared: dict[str, Any] = {}
    for split in data_files:
        if split not in ds:
            continue
        split_ds = ds[split]
        before = len(split_ds)
        split_ds = split_ds.filter(_audio_available, desc=f"Check audio {split}")
        dropped = before - len(split_ds)
        if dropped:
            logger.warning("[SKIP] %d/%d rows without audio in %s", dropped, before, split)
        if len(split_ds) == 0:
            logger.warning("[SKIP] Split %s is empty after audio check", split)
            continue

        def _map_fn(batch: dict[str, Any]) -> dict[str, Any]:
            audios = batch["audio"]
            texts = batch["transcript"]
            mel_batch: list[Any] = []
            label_batch: list[list[int]] = []
            duration_batch: list[float] = []
            length_batch: list[int] = []
            for audio, text in zip(audios, texts, strict=True):
                array, sr = load_clip(audio)
                encoded = encode_example(array, sr, str(text), processor)
                duration = float(len(array) / sr) if sr else 0.0
                full_mel = np.asarray(encoded["input_features"], dtype=np.float16)
                frames = max(1, min(round(duration * 100), full_mel.shape[1]))
                mel_batch.append(full_mel[:, :frames])
                label_batch.append(encoded["labels"])
                duration_batch.append(duration)
                length_batch.append(frames)
            return {
                "input_features": mel_batch,
                "labels": label_batch,
                "duration": duration_batch,
                "length": length_batch,
            }

        remove = [c for c in ("audio", "transcript") if c in split_ds.column_names]
        try:
            mapped = split_ds.map(
                _map_fn,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc if num_proc > 1 else None,
                remove_columns=remove,
                features=_precomputed_features(split_ds.features),
                desc=f"Precompute {split}",
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"Precompute failed on split {split}: {e}") from e
        prepared[split] = mapped

    if not prepared:
        raise ValueError(f"No splits prepared from {data_dir}")
    from datasets import DatasetDict

    output_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict(prepared).save_to_disk(str(output_dir))
    logger.info("Preprocessed dataset saved to %s", output_dir)
    return output_dir


def load_preprocessed_dataset(output_dir: Path) -> Any:
    """Load a precomputed Arrow DatasetDict with memory-mapping (never full RAM)."""
    from datasets import load_from_disk

    if not is_preprocessed(output_dir):
        raise FileNotFoundError(f"No preprocessed dataset at {output_dir}")
    try:
        return load_from_disk(str(output_dir), keep_in_memory=False)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Failed to load preprocessed dataset: {e}") from e
