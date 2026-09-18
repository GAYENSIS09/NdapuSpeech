"""Text normalization and Wolof corpus collection shared across the pipeline."""

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

_DOMAIN_MAPPING = {
    "kallaama": "agriculture",
    "alffa": "general",
    "fleurs": "general",
    "common_voice": "general",
    "wolbanking77": "banking",
    "radio": "radio",
    "health": "health",
    "family": "family",
    "taxi": "taxi",
    "market": "commerce",
    "school": "education",
    "admin": "administration",
    "business": "business",
    "interview": "interview",
    "discours": "religious",
}


def normalize_wolof_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace while removing '|' separators."""
    text = text.strip().lower()
    text = text.replace("|", " ")
    return " ".join(text.split())


def normalize_for_lm(text: str) -> str:
    """Normalize NFC and strip markdown/HTML markers for LM training."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_~`|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_domain(source: str) -> str:
    """Map a source identifier to a domain tag, falling back to 'other'."""
    for key, val in _DOMAIN_MAPPING.items():
        if key in source.lower():
            return val
    return "other"


def collect_text_corpora(paths: Sequence[Path]) -> list[str]:
    """Read text from existing files or *.txt files inside given directories."""
    texts: list[str] = []
    for path in paths:
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8"))
        elif path.is_dir():
            for txt_file in path.glob("*.txt"):
                texts.append(txt_file.read_text(encoding="utf-8"))
    return texts
