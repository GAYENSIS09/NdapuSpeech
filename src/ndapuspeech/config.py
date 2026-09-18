"""Central configuration for NdapuSpeech ASR project.

Config is loaded from ``configs/*.yaml`` at import time with silent fallback
to built-in defaults when a file is missing. Reading YAML is not a side effect;
directory creation must go through :func:`ensure_dirs`.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "configs"

_DATA: dict[str, dict[str, Any]] = {
    "config": {
        "paths": {
            "data_dir": "data",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "datasets_dir": "data/datasets",
            "models_dir": "models",
            "logs_dir": "logs",
        },
        "audio": {
            "target_sample_rate": 16000,
            "target_channels": 1,
            "vad_threshold": 0.5,
            "vad_min_speech_ms": 300,
            "vad_min_silence_ms": 100,
            "max_segment_seconds": 30,
            "min_segment_seconds": 3,
        },
        "model": {
            "base_model": "openai/whisper-large-v3-turbo",
            "pretrain_model": "openai/whisper-large-v3",
            "language": "wo",
            "language_tokens": ["<|wo|>", "<|wo-fam|>", "<|wo-formal|>"],
            "special_chars": ["ñ", "ë", "ŋ", "ɗ", "ɓ", "ƴ", "ó", "é", "à", "â", "ê"],
        },
    },
    "training": {
        "data_dir": "data/processed/ndapuspeech_ds",
        "output_dir": "models/ndapuspeech-v1",
        "learning_rate_decoder": 2.0e-5,
        "learning_rate_encoder": 1.0e-5,
        "warmup_steps": 100,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "max_epochs": 20,
        "early_stop_patience": 5,
        "train_split": 0.85,
        "seed": 42,
        "eval_steps": 200,
        "save_steps": 500,
        "use_lora": False,
        "resume_from": None,
        "max_steps": None,
        "noise_dir": None,
        "push_to_hub": None,
        "fp16": False,
        "bf16": False,
        "hub_dataset": "ndapuspeech/ndapuspeech-dataset",
    },
    "data": {
        "openslr_mirrors": [
            "https://www.openslr.org/resources",
            "https://mirror.indra.market/openslr/resources",
            "https://openslr.elda.org/resources",
        ],
        "hf_dataset_repo": "ndapuspeech/ndapuspeech-dataset",
        "hf_model_repo": "ndapuspeech/ndapuspeech-asr",
        "datasets": {
            "kallaama": {
                "resource_id": 151,
                "filename": "speech_dataset_wol.tar.gz",
                "fallback_url": (
                    "https://zenodo.org/records/10892569/files/speech_dataset_wol.tar.gz"
                ),
                "large": True,
            },
            "alffa": {
                "resource_id": 25,
                "filename": "data_readspeech_wo.tar.bz2",
                "large": True,
            },
            "fleurs": {"hub_id": "google/fleurs", "subset": "wo"},
            "wolbanking77": {"hub_id": "AI4D/WolBanking77_Speech"},
        },
        "include_sources": [
            "kallaama",
            "alffa",
            "fleurs",
            "wolbanking77",
        ],
        "manifests": [
            "data/processed/manifest.jsonl",
        ],
        "kenlm": {
            "corpora_paths": [
                "data/text/wolof",
                "data/raw/kallaama_text",
                "data/raw/wikipedia_wo.txt",
                "data/datasets/kallaama/wolof_text",
            ],
            "output_arpa": "models/kenlm/wolof_4gram.arpa",
            "output_bin": "models/kenlm/wolof_4gram.bin",
            "order": 4,
        },
        "tokenizer": {
            "vocab_size": 10000,
            "extended_tokenizer_dir": "models/whisper-tokenizer-extended",
            "corpora_paths": [
                "data/datasets/kallaama/wolof_text/wolof_text_corpus.txt",
                "data/raw/wikipedia_wo.txt",
                "data/raw/wolbanking77",
                "data/text/wolof",
            ],
        },
    },
    "youtube": {},
}

_PATH_LEAF_KEYS = {
    ("config", "paths", "data_dir"),
    ("config", "paths", "raw_data_dir"),
    ("config", "paths", "processed_data_dir"),
    ("config", "paths", "datasets_dir"),
    ("config", "paths", "models_dir"),
    ("config", "paths", "logs_dir"),
    ("training", "data_dir"),
    ("training", "output_dir"),
    ("training", "noise_dir"),
    ("training", "resume_from"),
    ("data", "manifests"),
    ("data", "kenlm", "corpora_paths"),
    ("data", "kenlm", "output_arpa"),
    ("data", "kenlm", "output_bin"),
    ("data", "tokenizer", "corpora_paths"),
    ("data", "tokenizer", "extended_tokenizer_dir"),
}


def _load_yaml(name: str) -> dict:
    """Load a YAML config file, returning {} when missing (fallback to defaults)."""
    path = CONFIG_DIR / name
    if not path.exists():
        logger.warning("[CONFIG] %s absent — using built-in defaults", path)
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RuntimeError(f"Malformed YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_path(value: object) -> Path | None:
    """Resolve a relative path against BASE_DIR; None stays None."""
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else BASE_DIR / p


def _resolve_paths(node: dict, prefix: tuple[str, ...]) -> dict:
    """Deep-resolve flagged path leaves in a config section against BASE_DIR."""
    out: dict = {}
    for key, value in node.items():
        path = (*prefix, key)
        if path in _PATH_LEAF_KEYS:
            if isinstance(value, list):
                out[key] = [_as_path(v) for v in value]
            else:
                out[key] = _as_path(value)
        elif isinstance(value, dict):
            out[key] = _resolve_paths(value, path)
        else:
            out[key] = value
    return out


def _build_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for name in ("config", "training", "data"):
        merged = _deep_merge(_DATA[name], _load_yaml(f"{name}.yaml"))
        sections.append(_resolve_paths(merged, (name,)))
    return sections[0], sections[1], sections[2]


CONFIG: dict[str, Any]
TRAINING: dict[str, Any]
DATA: dict[str, Any]

CONFIG, TRAINING, DATA = _build_config()

# ---- Paths (derived from config.yaml) ----
DATA_DIR: Path = CONFIG["paths"]["data_dir"]
RAW_DATA_DIR: Path = CONFIG["paths"]["raw_data_dir"]
PROCESSED_DATA_DIR: Path = CONFIG["paths"]["processed_data_dir"]
DATASETS_DIR: Path = CONFIG["paths"]["datasets_dir"]
MODELS_DIR: Path = CONFIG["paths"]["models_dir"]
LOGS_DIR: Path = CONFIG["paths"]["logs_dir"]

# ---- Audio ----
TARGET_SAMPLE_RATE: int = CONFIG["audio"]["target_sample_rate"]
TARGET_CHANNELS: int = CONFIG["audio"]["target_channels"]
VAD_THRESHOLD: float = CONFIG["audio"]["vad_threshold"]
VAD_MIN_SPEECH_MS: int = CONFIG["audio"]["vad_min_speech_ms"]
VAD_MIN_SILENCE_MS: int = CONFIG["audio"]["vad_min_silence_ms"]
MAX_SEGMENT_SECONDS: float = CONFIG["audio"]["max_segment_seconds"]
MIN_SEGMENT_SECONDS: float = CONFIG["audio"]["min_segment_seconds"]

# ---- Model ----
BASE_MODEL: str = CONFIG["model"]["base_model"]
PRETRAIN_MODEL: str = CONFIG["model"]["pretrain_model"]
LANGUAGE: str = CONFIG["model"]["language"]
WOLOF_SPEECH_TOKEN: str = CONFIG["model"]["language_tokens"][0]
FAMILY_TOKEN: str = CONFIG["model"]["language_tokens"][1]
FORMAL_TOKEN: str = CONFIG["model"]["language_tokens"][2]
WOLOF_SPECIAL_CHARS: list[str] = CONFIG["model"]["special_chars"]

# ---- Training (derived from training.yaml) ----
LEARNING_RATE_DECODER: float = TRAINING["learning_rate_decoder"]
LEARNING_RATE_ENCODER: float = TRAINING["learning_rate_encoder"]
WARMUP_STEPS: int = TRAINING["warmup_steps"]
BATCH_SIZE: int = TRAINING["batch_size"]
GRADIENT_ACCUMULATION: int = TRAINING["gradient_accumulation"]
MAX_EPOCHS: int = TRAINING["max_epochs"]
EARLY_STOP_PATIENCE: int = TRAINING["early_stop_patience"]
TRAIN_SPLIT: float = TRAINING["train_split"]
SEED: int = TRAINING["seed"]
TRAINING_DATA_DIR: Path | None = TRAINING["data_dir"]
TRAINING_OUTPUT_DIR: Path | None = TRAINING["output_dir"]
TRAINING_EVAL_STEPS: int = TRAINING["eval_steps"]
TRAINING_SAVE_STEPS: int = TRAINING["save_steps"]
TRAINING_USE_LORA: bool = TRAINING["use_lora"]
TRAINING_RESUME_FROM: Path | None = TRAINING["resume_from"]
TRAINING_MAX_STEPS: int | None = TRAINING["max_steps"]
TRAINING_NOISE_DIR: Path | None = TRAINING["noise_dir"]
TRAINING_PUSH_TO_HUB: str | None = TRAINING["push_to_hub"]
TRAINING_FP16: bool = TRAINING["fp16"]
TRAINING_BF16: bool = TRAINING["bf16"]

# ---- HF Hub ----
HF_DATASET_REPO: str = DATA["hf_dataset_repo"]
HF_MODEL_REPO: str = DATA["hf_model_repo"]
EXTENDED_TOKENIZER_DIR: Path = DATA["tokenizer"]["extended_tokenizer_dir"]

_DIRS_TO_CREATE: tuple[Path, ...] = (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    LOGS_DIR,
)


def ensure_dirs() -> None:
    """Create the project directory tree explicitly (never called at import)."""
    for d in _DIRS_TO_CREATE:
        d.mkdir(parents=True, exist_ok=True)


def get_config() -> dict:
    """Return the merged config sections (paths resolved) for display."""
    return {"config": CONFIG, "training": TRAINING, "data": DATA}
