"""KenLM n-gram language model training for beam-search rescoring."""

import logging
import subprocess
from pathlib import Path

from . import config
from .config import PROCESSED_DATA_DIR
from .text import collect_text_corpora, normalize_for_lm

logger = logging.getLogger(__name__)

_DEFAULT_CORPORA_PATHS = config.DATA["kenlm"]["corpora_paths"]


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

    texts = _load_corpus_texts(corpus_path)
    if not texts:
        raise FileNotFoundError("No Wolof text corpora found for KenLM training")

    normalized = "\n".join(normalize_for_lm(t) for t in texts)
    tmp = PROCESSED_DATA_DIR / "lm_corpus.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(normalized, encoding="utf-8")
    logger.info("Corpus: %s words", len(normalized.split()))

    logger.info("Training %d-gram KenLM...", order)
    lmplz_cmd = [
        "lmplz",
        "-o",
        str(order),
        "--text",
        str(tmp),
        "--arpa",
        str(arpa_path),
        "--discount_fallback",
    ]
    subprocess.run(lmplz_cmd, check=True)

    logger.info("Building binary LM...")
    build_binary_cmd = ["build_binary", str(arpa_path), str(bin_path)]
    result = subprocess.run(build_binary_cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("build_binary failed; keeping .arpa only: %s", result.stderr[:200])
        return arpa_path, None
    return arpa_path, bin_path
