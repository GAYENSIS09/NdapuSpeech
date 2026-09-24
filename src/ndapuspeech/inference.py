"""ASR inference: single-file transcription, KenLM rescoring, and batch decoding."""

import logging
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


def _load_pipeline(model_path: str) -> Any:
    """Build a HuggingFace ASR pipeline on the best available device."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )


def transcribe_file(model_path: str, audio_path: str, language: str = config.LANGUAGE) -> dict:
    """Transcribe a single audio file, returning the full pipeline result."""
    pipe = _load_pipeline(model_path)
    generate_kwargs = {"language": language, "task": "transcribe"}
    return pipe(audio_path, generate_kwargs=generate_kwargs, return_timestamps=True)


def transcribe_with_lm(model_path: str, audio_path: str, lm_path: str) -> dict:
    """Transcribe with KenLM beam-search rescoring over candidate sequences."""
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

    import kenlm

    lm = kenlm.Model(lm_path)

    audio, sr = librosa.load(audio_path, sr=config.TARGET_SAMPLE_RATE)
    input_features = processor.feature_extractor(
        audio, sampling_rate=sr, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            language=config.LANGUAGE,
            task="transcribe",
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=5,
            num_return_sequences=3,
        )

    candidates = [
        processor.decode(seq.cpu(), skip_special_tokens=True) for seq in outputs.sequences
    ]
    if not candidates:
        return {"text": "", "lm_score": None, "candidates": []}
    scored = sorted(
        ((cand, lm.score(cand, bos=True, eos=True)) for cand in candidates),
        key=lambda item: -item[1],
    )
    return {"text": scored[0][0], "lm_score": scored[0][1], "candidates": [c for c, _ in scored]}


def transcribe_batch(model_path: str, samples: list[dict], batch_size: int = 16) -> list[dict]:
    """Transcribe all samples, augmenting each with prediction and confidence."""
    pipe = _load_pipeline(model_path)

    results: list[dict] = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        audios = [b["audio"] for b in batch]
        outputs = pipe(audios, batch_size=batch_size, return_timestamps=False)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for sample, output in zip(batch, outputs, strict=True):
            copy = dict(sample)
            copy["prediction"] = output["text"].strip()
            copy["avg_logprob"] = output.get("avg_logprob", None)
            results.append(copy)
    return results


def transcribe_files(
    model_path: str,
    audio_paths: list[str],
    lm_path: str | None = None,
    language: str = config.LANGUAGE,
) -> dict[str, dict]:
    """Transcribe several audio files, optionally rescoring with a KenLM model."""
    transcriptions: dict[str, dict] = {}
    for audio_path in audio_paths:
        if not Path(audio_path).exists():
            logger.warning("[SKIP] %s not found", audio_path)
            continue
        if lm_path:
            result = transcribe_with_lm(model_path, audio_path, lm_path)
        else:
            result = transcribe_file(model_path, audio_path, language=language)
        transcriptions[audio_path] = {
            "text": result.get("text", ""),
            "avg_logprob": result.get("avg_logprob"),
            "segments": result.get("chunks"),
        }
    return transcriptions
