"""Audio preprocessing: VAD, resampling, normalization, long-form segmentation."""

import json
import logging
import subprocess
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    MAX_SEGMENT_SECONDS,
    MIN_SEGMENT_SECONDS,
    TARGET_SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_THRESHOLD,
)

logger = logging.getLogger(__name__)

_DEFAULT_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".mp4", ".m4a", ".webm", ".aac")


class AudioSegmenter:
    """Segment long audio into clean speech chunks suitable for ASR training."""

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        max_segment_sec: float = MAX_SEGMENT_SECONDS,
        min_segment_sec: float = MIN_SEGMENT_SECONDS,
    ):
        self.target_sr = target_sr
        self.max_segment_sec = max_segment_sec
        self.min_segment_sec = min_segment_sec
        self._vad_model: Any | None = None

    def _load_vad(self) -> Any:
        """Lazy-load the Silero VAD model."""
        if self._vad_model is None:
            import torch

            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                force_reload=False,
            )
            self._vad_model.eval()
        assert self._vad_model is not None
        return self._vad_model

    @staticmethod
    def load_audio(
        path: str, target_sr: int = TARGET_SAMPLE_RATE, mono: bool = True
    ) -> tuple[np.ndarray, int]:
        """Load audio at target sample rate using librosa."""
        import librosa

        return librosa.load(path, sr=target_sr, mono=mono)

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to the target sample rate if needed."""
        if orig_sr == self.target_sr:
            return audio
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def normalize(
        self, audio: np.ndarray, peak: float = 0.99, db: float | None = None
    ) -> np.ndarray:
        """Normalize loudness. If db given, use relative LUFS normalization."""
        if db is not None:
            import pyloudnorm as pyln

            meter = pyln.Meter(self.target_sr)
            loudness = meter.integrated_loudness(audio)
            return pyln.normalize.loudness(audio, loudness, db)
        if np.max(np.abs(audio)) > 0:
            return audio * (peak / np.max(np.abs(audio)))
        return audio

    def detect_speech_blocks(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """Detect speech segments with Silero VAD, returning (start_sec, end_sec) pairs."""
        import torch

        model = self._load_vad()
        sr = self.target_sr

        window_size = 512
        n_chunks = len(audio) // window_size
        speech_chunks = np.zeros(n_chunks, dtype=bool)

        audio_tensor = torch.from_numpy(audio[: n_chunks * window_size]).float()

        with torch.no_grad():
            for i in range(0, len(audio_tensor) - window_size + 1, window_size):
                chunk = audio_tensor[i : i + window_size]
                prob = model(chunk, sr).item()
                speech_chunks[i // window_size] = prob > VAD_THRESHOLD

        blocks: list[tuple[float, float]] = []
        in_speech = False
        start_idx = 0
        for i, is_speech in enumerate(speech_chunks):
            if is_speech and not in_speech:
                in_speech = True
                start_idx = i
            elif not is_speech and in_speech:
                in_speech = False
                blocks.append((start_idx * window_size / sr, i * window_size / sr))
        if in_speech:
            blocks.append((start_idx * window_size / sr, len(speech_chunks) * window_size / sr))

        min_dur = VAD_MIN_SPEECH_MS / 1000
        blocks = [b for b in blocks if b[1] - b[0] >= min_dur]

        merged: list[tuple[float, float]] = []
        for block in blocks:
            if merged and block[0] - merged[-1][1] < VAD_MIN_SILENCE_MS / 1000:
                merged[-1] = (merged[-1][0], block[1])
            else:
                merged.append(block)
        return merged

    @staticmethod
    def split_long_block(
        block: tuple[float, float], max_segment_sec: float
    ) -> list[tuple[float, float]]:
        """Split a long speech block into fixed-length chunks."""
        dur = block[1] - block[0]
        if dur <= max_segment_sec:
            return [block]
        segments = []
        start = block[0]
        while start < block[1]:
            end = min(start + max_segment_sec, block[1])
            segments.append((start, end))
            start = end
        return segments

    def segment_audio(
        self, audio_path: str, out_dir: Path, prefix: str = "", min_db: float = -35.0
    ) -> list[dict]:
        """Load, VAD-segment, split, and save segments with metadata."""
        import soundfile as sf

        audio, sr = self.load_audio(audio_path, target_sr=None)
        audio = self.normalize(audio)
        audio = self.resample(audio, sr)

        rms = np.sqrt(np.mean(audio**2)) if len(audio) else 0.0
        db = 20 * np.log10(rms + 1e-10)
        if db < min_db:
            logger.info("[SKIP] %s: too quiet (%.1f dB)", audio_path, db)
            return []

        blocks = self.detect_speech_blocks(audio)

        segments: list[dict] = []
        for i, block in enumerate(blocks):
            for j, sub in enumerate(self.split_long_block(block, self.max_segment_sec)):
                start_s, end_s = sub
                seg_audio = audio[int(start_s * self.target_sr) : int(end_s * self.target_sr)]
                if len(seg_audio) / self.target_sr < self.min_segment_sec:
                    continue

                fname = f"{prefix}{i:04d}_{j:02d}.wav" if prefix else f"{i:04d}_{j:02d}.wav"
                out_path = out_dir / fname
                sf.write(str(out_path), seg_audio, self.target_sr)
                segments.append(
                    {
                        "file": str(out_path),
                        "duration": round(len(seg_audio) / self.target_sr, 2),
                        "start": round(start_s, 2),
                        "end": round(end_s, 2),
                        "source": str(audio_path),
                    }
                )
        return segments


def convert_to_wav(
    input_path: Path, output_path: Path, sample_rate: int = TARGET_SAMPLE_RATE
) -> Path:
    """Convert any audio/video file to 16kHz mono WAV via ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def make_manifest(segments: list[dict], out_path: Path) -> None:
    """Save segment metadata as a JSONL manifest."""
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def process_directory(
    input_dir: Path, output_dir: Path, audio_exts: tuple[str, ...] | None = None
) -> list[dict]:
    """Segment every audio file under input_dir (recursively) into output_dir."""
    audio_exts = audio_exts or _DEFAULT_AUDIO_EXTS
    segmenter = AudioSegmenter()
    all_segments: list[dict] = []

    files = [
        Path(p) for ext in audio_exts for p in glob(str(input_dir / f"**/*{ext}"), recursive=True)
    ]
    for audio_file in files:
        try:
            all_segments.extend(segmenter.segment_audio(str(audio_file), output_dir))
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("[ERR] %s: %s", audio_file, e)

    make_manifest(all_segments, output_dir.parent / "manifest.jsonl")
    return all_segments
