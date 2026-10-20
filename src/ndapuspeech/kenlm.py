"""KenLM n-gram language model training for beam-search rescoring."""

import logging
import re
import shutil
import subprocess
from pathlib import Path

from . import config
from .config import PROCESSED_DATA_DIR
from .text import collect_text_corpora, normalize_for_lm

logger = logging.getLogger(__name__)

_DEFAULT_CORPORA_PATHS = config.DATA["kenlm"]["corpora_paths"]

_INSTALL_HINT = (
    "KenLM C++ tools ('lmplz', 'build_binary') not found on PATH. "
    "On Windows build from https://github.com/kpu/kenlm (or train inside WSL: "
    "'sudo apt install build-essential cmake && mkdir build && cd build && cmake .. && make -j'). "
    "KenLM is optional (inference rescoring only); skip this step to train Whisper."
)


def _require_binary(name: str) -> str:
    """Return the resolved path of a KenLM tool, raising a typed error if absent."""
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"{_INSTALL_HINT} (missing: {name})")
    return found


def _to_sentences(text: str) -> list[str]:
    """Split normalized text into non-empty sentences on end punctuation."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _load_corpus_texts(corpus_path: Path | None) -> list[str]:
    """Gather raw corpus texts from an explicit path or default locations."""
    if corpus_path is not None:
        if corpus_path.is_dir():
            return [f.read_text(encoding="utf-8") for f in corpus_path.glob("*.txt")]
        return [corpus_path.read_text(encoding="utf-8")]
    return collect_text_corpora(_DEFAULT_CORPORA_PATHS)


def train_kenlm(
    corpus_path: Path | None = None,
    output_arpa: Path | None = None,
    output_bin: Path | None = None,
    order: int = config.DATA["kenlm"]["order"],
) -> tuple[Path, Path | None]:
    """Train a KenLM n-gram model, converting the ARPA output to a binary if possible.

    Paths default to the values from ``configs/data.yaml``. Returns (arpa, bin) paths.
    """
    arpa_path = output_arpa or config.DATA["kenlm"]["output_arpa"]
    bin_path = output_bin or config.DATA["kenlm"]["output_bin"]
    arpa_path.parent.mkdir(parents=True, exist_ok=True)

    lmplz = _require_binary("lmplz")

    texts = _load_corpus_texts(corpus_path)
    if not texts:
        raise FileNotFoundError("No Wolof text corpora found for KenLM training")

    lines = [s for t in texts for s in _to_sentences(normalize_for_lm(t))]
    if not lines:
        raise FileNotFoundError("No usable sentences found for KenLM training")
    normalized = "\n".join(lines)
    tmp = PROCESSED_DATA_DIR / "lm_corpus.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(normalized, encoding="utf-8")
    logger.info("Corpus: %s words in %s sentences", len(normalized.split()), len(lines))

    logger.info("Training %d-gram KenLM...", order)
    lmplz_cmd = [
        lmplz,
        "-o",
        str(order),
        "--text",
        str(tmp),
        "--arpa",
        str(arpa_path),
        "--discount_fallback",
    ]
    try:
        subprocess.run(lmplz_cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{_INSTALL_HINT} (missing: lmplz)") from e

    logger.info("Building binary LM...")
    try:
        build_binary = _require_binary("build_binary")
    except RuntimeError:
        logger.warning("build_binary absent; keeping .arpa only")
        return arpa_path, None
    build_binary_cmd = [build_binary, str(arpa_path), str(bin_path)]
    result = subprocess.run(build_binary_cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("build_binary failed; keeping .arpa only: %s", result.stderr[:200])
        return arpa_path, None
    return arpa_path, bin_path
