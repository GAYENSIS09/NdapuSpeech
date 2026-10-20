"""ASR inference: single-file transcription, KenLM rescoring, and batch decoding."""

import logging
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


def _supported_languages() -> set[str]:
    """Return Whisper language codes supported by the installed transformers."""
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
    except ImportError:
        try:
            from transformers import WhisperTokenizer

            langs = set(getattr(WhisperTokenizer, "SUPPORTED_LANGUAGES", ()))
            return set(langs) if langs else set()
        except ImportError:
            return set()
    return set(LANGUAGES.keys())


def _sanitize_language(language: str | None) -> str | None:
    """Map unsupported codes (e.g. Wolof 'wo') to auto-detection (None).

    Whisper has no Wolof token; forcing ``language='wo'`` raises
    ``ValueError: Unsupported language``. Fine-tuned NdapuSpeech models
    transcribe Wolof without language conditioning, so we fall back to
    auto-detection with a warning instead of crashing.
    """
    if language is None:
        return None
    supported = _supported_languages()
    if supported and language not in supported:
        logger.warning(
            "[LANG] '%s' unsupported by Whisper — using auto-detection", language
        )
        return None
    return language


def _generate_kwargs(language: str | None) -> dict:
    """Build generate kwargs, omitting unsupported languages.

    NdapuSpeech is fine-tuned in free-transcription mode (no language token):
    when the requested language is unsupported, return no language/task at all
    so generation starts from ``[<|startoftranscript|>, <|notimestamps|>]``.
    """
    clean = _sanitize_language(language)
    if clean is None:
        return {}
    return {"language": clean, "task": "transcribe"}


def _load_pipeline(model_path: str, force_cpu: bool = False) -> Any:
    """Build a HuggingFace ASR pipeline on the best available device."""
    import torch
    from transformers import pipeline

    from .models import configure_free_transcription

    use_cuda = torch.cuda.is_available() and not force_cpu
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        dtype=dtype,
    )
    configure_free_transcription(pipe.model)
    return pipe


def transcribe_file(
    model_path: str, audio_path: str, language: str | None = config.LANGUAGE
) -> dict:
    """Transcribe a single audio file, returning the full pipeline result."""
    import torch

    pipe = _load_pipeline(model_path)
    try:
        try:
            return pipe(
                audio_path,
                generate_kwargs=_generate_kwargs(language),
                return_timestamps=True,
            )
        except Exception as e:
            if not _is_cuda_error(e):
                raise
            logger.warning("[CUDA] single-file GPU failed (%s); retry on CPU", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del pipe
            cpu_pipe = _load_pipeline(model_path, force_cpu=True)
            return cpu_pipe(
                audio_path,
                generate_kwargs=_generate_kwargs(language),
                return_timestamps=True,
            )
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def transcribe_with_lm(model_path: str, audio_path: str, lm_path: str) -> dict:
    """Transcribe with KenLM beam-search rescoring over candidate sequences."""
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    from .models import configure_free_transcription

    configure_free_transcription(model)

    try:
        import kenlm
    except ImportError as e:
        raise RuntimeError("pip install kenlm for LM rescoring (or pass lm_path=None)") from e

    lm = kenlm.Model(lm_path)

    audio, sr = librosa.load(audio_path, sr=config.TARGET_SAMPLE_RATE)
    input_features = processor.feature_extractor(
        audio, sampling_rate=sr, return_tensors="pt"
    ).to(device)

    gen_kwargs: dict = {"num_beams": 5, "num_return_sequences": 3}

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
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


def _is_oom(exc: BaseException) -> bool:
    """Return True for any CUDA OOM flavour (OutOfMemoryError, AcceleratorError...)."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "out of memory" in msg or "cudamalloc" in msg or "oom" in name


def _is_cuda_error(exc: BaseException) -> bool:
    """Return True for any CUDA runtime failure worth retrying on CPU."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        _is_oom(exc)
        or "cuda" in msg
        or "cudnn" in msg
        or "cublas" in msg
        or "accelerator" in name
    )


def transcribe_batch(model_path: str, samples: list[dict], batch_size: int = 16) -> list[dict]:
    """Transcribe all samples, augmenting each with prediction and confidence."""
    import torch

    pipe = _load_pipeline(model_path)

    def _decode_one(p: Any, audio_ref: str) -> dict:
        out = p(audio_ref, batch_size=1, return_timestamps=False)
        if isinstance(out, list):
            return out[0]
        return out

    def _run(p: Any) -> list[dict]:
        out_all: list[dict] = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            audios = [b["audio"] for b in batch]
            try:
                outputs = p(audios, batch_size=batch_size, return_timestamps=False)
            except Exception as e:
                if not _is_cuda_error(e):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("[CUDA] batch %d retried sample-by-sample (%s)", i, e)
                outputs = [_decode_one(p, a) for a in audios]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not isinstance(outputs, list):
                outputs = [outputs]
            for sample, output in zip(batch, outputs, strict=True):
                copy = dict(sample)
                copy["prediction"] = output["text"].strip()
                copy["avg_logprob"] = output.get("avg_logprob", None)
                out_all.append(copy)
        return out_all

    try:
        results = _run(pipe)
    except Exception as e:
        if not _is_cuda_error(e):
            raise
        logger.warning("[CUDA] GPU failed (%s); retrying full batch on CPU", e)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        del pipe
        cpu_pipe = _load_pipeline(model_path, force_cpu=True)
        results = _run(cpu_pipe)
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return results


def transcribe_files(
    model_path: str,
    audio_paths: list[str],
    lm_path: str | None = None,
    language: str | None = config.LANGUAGE,
) -> dict[str, dict]:
    """Transcribe several audio files, optionally rescoring with a KenLM model."""
    transcriptions: dict[str, dict] = {}
    for audio_path in audio_paths:
        if not Path(audio_path).exists():
            logger.warning("[SKIP] %s not found", audio_path)
            continue
        try:
            if lm_path:
                result = transcribe_with_lm(model_path, audio_path, lm_path)
            else:
                result = transcribe_file(model_path, audio_path, language=language)
        except RuntimeError as e:
            logger.warning("[FALLBACK] LM rescoring failed (%s); plain decode", e)
            result = transcribe_file(model_path, audio_path, language=language)
        transcriptions[audio_path] = {
            "text": result.get("text", ""),
            "avg_logprob": result.get("avg_logprob"),
            "segments": result.get("chunks"),
        }
    return transcriptions
