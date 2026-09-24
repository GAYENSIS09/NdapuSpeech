"""Dataset download and unified dataset construction for NdapuSpeech ASR."""

import json
import logging
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config
from .config import RAW_DATA_DIR
from .text import detect_domain, normalize_wolof_text

logger = logging.getLogger(__name__)

OPENSLR_MIRRORS = config.DATA["openslr_mirrors"]


def _is_nonempty(path: Path) -> bool:
    """Return True if the file exists and contains at least one byte."""
    return path.exists() and path.stat().st_size > 0


def _has_files(path: Path) -> bool:
    """Return True when a directory contains at least one non-empty file."""
    return path.is_dir() and any(p.is_file() and _is_nonempty(p) for p in path.rglob("*"))


def _is_saved_hf_dataset(path: Path) -> bool:
    """Return True when a Hugging Face Dataset or DatasetDict is saved locally."""
    return (path / "dataset_dict.json").is_file() or any(
        p.is_file() for p in path.glob("*/dataset_info.json")
    )


def _is_valid_archive(path: Path) -> bool:
    """Return True when a tar archive can be opened and read."""
    if not _is_nonempty(path) or not path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return False
    try:
        with tarfile.open(path, "r:*") as archive:
            archive.getmembers()
    except (OSError, tarfile.TarError):
        return False
    return True


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with curl, falling back to aria2c."""
    if _is_valid_archive(dest_path) or (
        _is_nonempty(dest_path) and not dest_path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
    ):
        logger.info("[SKIP] %s already exists", dest_path.name)
        return dest_path
    logger.info("[DL] %s", url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = dest_path.with_name(f".{dest_path.name}.part")
    partial_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["curl", "--fail", "-L", "-o", str(partial_path), "--retry", "3", url],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        partial_path.unlink(missing_ok=True)
        subprocess.run(
            [
                "aria2c",
                "-x",
                "16",
                "-s",
                "16",
                "-d",
                str(partial_path.parent),
                "-o",
                partial_path.name,
                url,
            ],
            check=True,
        )
    if not _is_nonempty(partial_path):
        partial_path.unlink(missing_ok=True)
        raise OSError(f"Downloaded file is empty: {url}")
    partial_path.replace(dest_path)
    return dest_path


def try_mirrors(resource_id: int, filename: str) -> Path:
    """Try downloading a file from multiple OpenSLR mirrors."""
    for mirror in OPENSLR_MIRRORS:
        url = f"{mirror}/{resource_id}/{filename}"
        dest = RAW_DATA_DIR / filename
        try:
            download_file(url, dest)
            if _is_nonempty(dest):
                return dest
        except (subprocess.CalledProcessError, OSError) as e:
            logger.error("[ERR] Mirror %s failed: %s", mirror, e)
            continue
    raise RuntimeError(f"All mirrors failed for {filename}")


def _discard_corrupt_archive(archive_path: Path) -> None:
    """Delete a corrupt/truncated download so the next attempt re-downloads it."""
    try:
        archive_path.unlink(missing_ok=True)
        logger.warning("[CLEAN] Removed corrupt archive %s", archive_path.name)
    except OSError:
        logger.warning("[WARN] Could not remove corrupt archive %s", archive_path.name)


def extract_tar(archive_path: Path, dest_dir: Path) -> Path:
    """Extract a tar archive to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[EXTRACT] %s -> %s", archive_path.name, dest_dir)
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(dest_dir, filter="data")
    except (tarfile.TarError, EOFError) as e:
        _discard_corrupt_archive(archive_path)
        raise RuntimeError(f"Failed to extract {archive_path} (truncated download?): {e}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to extract {archive_path}: {e}") from e
    return dest_dir


def download_kallaama() -> Path:
    """Download the Kallaama Wolof speech dataset (55h, agriculture)."""
    logger.info("=== Kallaama (OpenSLR 151) — 55h agriculture ===")
    dest = RAW_DATA_DIR / "kallaama"
    ds_cfg = config.DATA["datasets"]["kallaama"]
    if _has_files(dest):
        logger.info("[SKIP] Kallaama already exists at %s", dest)
        return dest
    try:
        archive = try_mirrors(ds_cfg["resource_id"], ds_cfg["filename"])
    except RuntimeError:
        archive = download_file(ds_cfg["fallback_url"], RAW_DATA_DIR / ds_cfg["filename"])
    extract_tar(archive, dest)
    logger.info("[OK] Kallaama extracted to %s", dest)
    return dest


def download_alffa() -> Path:
    """Download the ALFFA Wolof read speech dataset (18h)."""
    logger.info("=== ALFFA (OpenSLR 25) — 18h read speech ===")
    dest = RAW_DATA_DIR / "alffa"
    ds_cfg = config.DATA["datasets"]["alffa"]
    if _has_files(dest):
        logger.info("[SKIP] ALFFA already exists at %s", dest)
        return dest
    archive = try_mirrors(ds_cfg["resource_id"], ds_cfg["filename"])
    extract_tar(archive, dest)
    logger.info("[OK] ALFFA extracted to %s", dest)
    return dest


def _download_hf_dataset(
    hub_id: str, out_name: str, subsets: list[str] | None = None
) -> Path | None:
    """Download a HuggingFace dataset and save it to disk."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets first")
        return None
    out = RAW_DATA_DIR / out_name
    if _is_saved_hf_dataset(out):
        logger.info("[SKIP] %s already exists at %s", out_name, out)
        return out
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
        logger.warning("[CLEAN] Removed partial dataset dir %s", out.name)
    ds = load_dataset(
        hub_id,
        name=subsets[0] if subsets else None,
    )
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    logger.info("[OK] %s saved to %s", out_name, out)
    return out


def download_fleurs() -> Path | None:
    """Download Google FLEURS Wolof via HuggingFace."""
    logger.info("=== Google FLEURS Wolof — ~10h multi-domain ===")
    ds_cfg = config.DATA["datasets"]["fleurs"]
    return _download_hf_dataset(ds_cfg["hub_id"], "fleurs", subsets=[ds_cfg["subset"]])


def download_wolbanking77() -> Path | None:
    """Download WolBanking77 Wolof banking speech (4h) via HuggingFace."""
    logger.info("=== WolBanking77 — 4h banking ===")
    ds_cfg = config.DATA["datasets"]["wolbanking77"]
    return _download_hf_dataset(ds_cfg["hub_id"], "wolbanking77")


def download_all(sources: list[str] | None = None) -> dict[str, Path | None]:
    """Download all configured datasets (or a subset), returning their dirs."""
    active_sources = sources if sources is not None else config.DATA["include_sources"]
    handlers: dict[str, Callable[[], Path | None]] = {
        "kallaama": download_kallaama,
        "alffa": download_alffa,
        "fleurs": download_fleurs,
        "wolbanking77": download_wolbanking77,
    }
    results: dict[str, Path | None] = {}
    for source in active_sources:
        handler = handlers.get(source)
        if handler is None:
            logger.warning("No downloader for source %r", source)
            continue
        for attempt in range(2):
            try:
                results[source] = handler()
                break
            except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
                if attempt == 0:
                    logger.warning("[RETRY] %s download failed (%s), retrying once", source, e)
                    continue
                logger.error("[ERR] %s download failed: %s", source, e)
                results[source] = None
    return results


def load_jsonl(path: Path) -> list[dict]:
    """Load samples from a JSONL file, skipping empty and invalid lines."""
    samples: list[dict] = []
    if not path.exists():
        return samples
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def build_from_manifest(manifest_path: Path, domain: str = "youtube") -> list[dict]:
    """Convert a JSONL manifest into unified samples, keeping valid transcriptions only."""
    samples: list[dict] = []
    for seg in load_jsonl(manifest_path):
        transcript = seg.get("transcript", "").strip()
        audio_path = seg.get("file", "")
        if not audio_path or not transcript:
            continue
        if not Path(audio_path).exists():
            continue
        samples.append(
            {
                "audio": str(audio_path),
                "transcript": normalize_wolof_text(transcript),
                "domain": seg.get("domain", domain),
                "source": seg.get("source", domain),
                "duration": seg.get("duration", 0.0),
                "split": "",
            }
        )
    return samples


def build_from_alffa(ds_dir: Path) -> list[dict]:
    """Parse ALFFA dataset in Kaldi format (text + wav files in speaker dirs)."""
    samples: list[dict] = []
    data_dir = ds_dir / "data_readspeech_wo" / "data"
    if not data_dir.exists():
        logger.warning("[WARN] ALFFA data dir not found at %s", data_dir)
        return samples
    import soundfile as sf

    for split_name in ("train", "dev", "test"):
        split_dir = data_dir / split_name
        text_file = split_dir / "text"
        if not text_file.exists():
            logger.warning("[WARN] Missing text file for ALFFA %s", split_name)
            continue
        # Load text: utt_id -> transcript
        with text_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                # utt_id format: WOL_01_lect_0001 -> speaker=01, find wav in split_dir/01/
                speaker_id = utt_id.split("_")[1]  # "01"
                wav_path = split_dir / speaker_id / f"{utt_id}.wav"
                if not wav_path.exists():
                    # try alternative naming
                    alt_wav = split_dir / speaker_id / f"{utt_id}.WAV"
                    if alt_wav.exists():
                        wav_path = alt_wav
                    else:
                        continue
                # Get duration from audio file
                try:
                    info = sf.info(str(wav_path))
                    duration = info.duration
                except Exception:
                    duration = 0.0
                samples.append(
                    {
                        "audio": str(wav_path.resolve()),
                        "transcript": normalize_wolof_text(transcript),
                        "domain": "read_speech",
                        "source": "alffa",
                        "duration": duration,
                        "split": split_name,
                    }
                )
    return samples


def build_from_kallaama(ds_dir: Path) -> list[dict]:
    """Parse Kallaama dataset in STM format and extract audio segments."""
    samples: list[dict] = []
    speech_dir = ds_dir / "clean_dataset_ready4release" / "wolof" / "speech_dataset"
    if not speech_dir.exists():
        logger.warning("[WARN] Kallaama speech dir not found at %s", speech_dir)
        return samples
    import re

    import soundfile as sf

    # STM format: recording_id channel speaker start end <tags> transcript
    stm_pattern = re.compile(
        r"^(\S+)\s+(\d+)\s+(\S+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+<[^>]+>\s+(.+)$"
    )
    segments_dir = ds_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    for speaker_dir in speech_dir.iterdir():
        if not speaker_dir.is_dir() or speaker_dir.name == "checked_transcriptions":
            continue
        stm_dir = speaker_dir / "stm_format"
        if not stm_dir.exists():
            continue
        for stm_file in stm_dir.glob("*.stm"):
            audio_file = speaker_dir / f"{stm_file.stem}.wav"
            if not audio_file.exists():
                continue
            # Load full audio once
            try:
                audio_data, sr = sf.read(str(audio_file))
            except Exception as e:
                logger.warning("[WARN] Failed to read %s: %s", audio_file, e)
                continue
            with stm_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";;"):
                        continue
                    m = stm_pattern.match(line)
                    if not m:
                        continue
                    rec_id, channel, speaker, start_str, end_str, transcript = m.groups()
                    start = float(start_str)
                    end = float(end_str)
                    duration = end - start
                    if duration < 1.0:
                        continue
                    # Clean transcript
                    clean_transcript = re.sub(r"%\w+|\[[^\]]+\]|<[^>]+>", "", transcript)
                    clean_transcript = re.sub(r"\s+", " ", clean_transcript).strip()
                    if not clean_transcript:
                        continue
                    # Unique segment ID
                    seg_id = f"{rec_id}_{channel}_{speaker}_{start_str}_{end_str}"
                    # Extract segment
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    if end_sample > len(audio_data):
                        end_sample = len(audio_data)
                    segment_audio = audio_data[start_sample:end_sample]
                    # Save segment
                    segment_path = segments_dir / f"{seg_id}.wav"
                    try:
                        sf.write(str(segment_path), segment_audio, sr)
                    except Exception as e:
                        logger.warning("[WARN] Failed to write segment %s: %s", segment_path, e)
                        continue
                    samples.append(
                        {
                            "audio": str(segment_path.resolve()),
                            "transcript": normalize_wolof_text(clean_transcript),
                            "domain": "radio",
                            "source": "kallaama",
                            "duration": duration,
                            "split": "",
                        }
                    )
    return samples


def build_from_dataset_dir(ds_dir: Path) -> list[dict]:
    """Convert a HuggingFace dataset saved to disk into unified samples."""
    samples: list[dict] = []
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError:
        logger.warning("pip install datasets to build from dataset dir")
        return samples
    ds = load_from_disk(str(ds_dir))
    if not isinstance(ds, DatasetDict):
        return samples
    for split_name, split_ds in ds.items():
        for ex in split_ds:
            if not isinstance(ex, dict):
                continue
            audio = ex.get("audio")
            audio_path = ""
            if isinstance(audio, dict):
                audio_path = audio.get("path", "")
            elif isinstance(audio, str):
                audio_path = audio
            text = ex.get("sentence") or ex.get("transcription") or ex.get("text")
            if not audio_path or not text:
                continue
            if not Path(audio_path).exists():
                continue
            samples.append(
                {
                    "audio": str(audio_path),
                    "transcript": normalize_wolof_text(str(text)),
                    "domain": detect_domain(ds_dir.name),
                    "source": ds_dir.name,
                    "duration": ex.get("duration", 0.0),
                    "split": split_name,
                }
            )
    return samples


def deduplicate_by_audio(samples: list[dict]) -> list[dict]:
    """Remove samples sharing the same resolved audio path."""
    seen: set[Path] = set()
    unique: list[dict] = []
    for sample in samples:
        key = Path(sample["audio"]).resolve()
        if key not in seen:
            seen.add(key)
            unique.append(sample)
    return unique


def split_dataset(
    samples: list[dict], train_ratio: float = config.TRAIN_SPLIT, seed: int = config.SEED
) -> list[dict]:
    """Stratify by source and split into train/val/test with a fixed seed."""
    import random

    random.seed(seed)
    by_source: dict[str, list[dict]] = {}
    for s in samples:
        by_source.setdefault(s["source"], []).append(s)

    result: list[dict] = []
    for group in by_source.values():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * 0.10)
        ordered = sorted(group, key=lambda x: x.get("duration", 0), reverse=True)
        for i, s in enumerate(ordered):
            if i < n_train:
                s["split"] = "train"
            elif i < n_train + n_val:
                s["split"] = "val"
            else:
                s["split"] = "test"
            decoded: dict = {k: v for k, v in s.items()}
            if decoded["audio"].startswith("data"):
                decoded["audio"] = str((Path(config.BASE_DIR) / decoded["audio"]).resolve())
            result.append(decoded)
    return result


def build_from_hf_dataset_dir(ds_dir: Path) -> list[dict]:
    """Parse a HuggingFace dataset saved to disk (metadata only, no audio decoding)."""
    samples: list[dict] = []
    try:
        from datasets import Audio, DatasetDict, load_from_disk
    except ImportError:
        logger.warning("pip install datasets to build from dataset dir")
        return samples
    try:
        ds = load_from_disk(str(ds_dir))
    except Exception as e:
        logger.warning("[WARN] Failed to load HF dataset from %s: %s", ds_dir, e)
        return samples
    if not isinstance(ds, DatasetDict):
        return samples
    # Directory to save extracted audio files
    audio_out_dir = ds_dir / "extracted_audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in ds.items():
        # Remove audio feature decoding
        if "audio" in split_ds.column_names:
            split_ds = split_ds.cast_column("audio", Audio(decode=False))
        for ex in split_ds:
            if not isinstance(ex, dict):
                continue
            audio = ex.get("audio")
            audio_path = ""
            if isinstance(audio, dict):
                # Audio stored as bytes inline - extract and save
                audio_bytes = audio.get("bytes")
                rel_path = audio.get("path", "audio.wav")
                if audio_bytes:
                    # Save to local file
                    save_path = audio_out_dir / f"{split_name}_{rel_path}"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # Write bytes to file
                        with open(save_path, "wb") as f:
                            f.write(audio_bytes)
                        audio_path = str(save_path.resolve())
                    except Exception:
                        continue
                elif audio.get("path"):
                    audio_path = audio.get("path", "")
            elif isinstance(audio, str):
                audio_path = audio
            text = ex.get("sentence") or ex.get("transcription") or ex.get("text")
            if not audio_path or not text:
                continue
            if not Path(audio_path).exists():
                continue
            # Compute duration if not present (FLEURS uses num_samples at 16kHz)
            duration = ex.get("duration", 0.0)
            if duration == 0.0 and ex.get("num_samples"):
                duration = ex["num_samples"] / 16000.0
            samples.append(
                {
                    "audio": str(Path(audio_path).resolve()),
                    "transcript": normalize_wolof_text(str(text)),
                    "domain": detect_domain(ds_dir.name),
                    "source": ds_dir.name,
                    "duration": duration,
                    "split": split_name,
                }
            )
    return samples


def build_unified_dataset(
    output_dir: Path | str | None = None,
    train_ratio: float = config.TRAIN_SPLIT,
    push_to_hub: str | None = None,
) -> dict[str, Any]:
    """Merge all sources into a deduplicated train/val/test JSONL dataset.

    Returns a summary dict (samples, hours per source, splits).
    """
    out_dir = Path(output_dir) if output_dir else config.PROCESSED_DATA_DIR / "ndapuspeech_ds"
    # Check if dataset already exists (summary.json + splits)
    summary_path = out_dir / "summary.json"
    splits_exist = all((out_dir / f"{split}.jsonl").exists() for split in ("train", "val", "test"))
    if summary_path.exists() and splits_exist:
        logger.info("[SKIP] Unified dataset already exists at %s", out_dir)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    all_samples: list[dict] = []

    for manifest in config.DATA["manifests"]:
        samples = build_from_manifest(manifest)
        if not samples:
            continue
        source = manifest.parent.parent.name
        for s in samples:
            s["source"] = source
        all_samples.extend(samples)

    for ds_dir in config.RAW_DATA_DIR.glob("*"):
        if not ds_dir.is_dir():
            continue
        if ds_dir.name.startswith("alffa"):
            all_samples.extend(build_from_alffa(ds_dir))
        elif ds_dir.name.startswith("kallaama"):
            all_samples.extend(build_from_kallaama(ds_dir))
        elif ds_dir.name in ("fleurs", "common_voice", "wolbanking77"):
            all_samples.extend(build_from_hf_dataset_dir(ds_dir))
        elif any(ds_dir.name.startswith(x) for x in config.DATA["include_sources"]):
            all_samples.extend(build_from_dataset_dir(ds_dir))

    unique = deduplicate_by_audio(all_samples)
    total_hours = sum(s.get("duration", 0) or 0 for s in unique) / 3600

    by_source: dict[str, float] = {}
    for s in unique:
        by_source[s["source"]] = by_source.get(s["source"], 0.0) + s.get("duration", 0) / 3600

    final = split_dataset(unique, train_ratio=train_ratio)
    splits: dict[str, list[dict]] = {}
    for s in final:
        splits.setdefault(s["split"], []).append(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for s in splits.get(split, []):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if push_to_hub:
        from datasets import Audio, load_dataset

        data_files = {split: str(out_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
        ds = load_dataset("json", data_files=data_files)
        ds = ds.cast_column("audio", Audio(sampling_rate=config.TARGET_SAMPLE_RATE))
        ds.push_to_hub(push_to_hub)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "total_hours": total_hours,
        "n_samples": len(unique),
        "by_source": by_source,
        "splits": {k: len(v) for k, v in splits.items()},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
