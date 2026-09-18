"""WER/CER metrics and per-domain evaluation for NdapuSpeech."""

import logging
from collections import defaultdict

from .inference import transcribe_batch

logger = logging.getLogger(__name__)


def _jiwer(fn_name: str, predictions: list[str], references: list[str]) -> float:
    """Run a jiwer metric by name, guarding against unavailable package."""
    import jiwer

    fn = getattr(jiwer, fn_name)
    return fn(references, predictions)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Word error rate, 0.0 = perfect."""
    return _jiwer("wer", predictions, references)


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Character error rate, 0.0 = perfect."""
    return _jiwer("cer", predictions, references)


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Return WER/CER/MER/WIL percentages plus sample count."""
    metrics = {
        "wer": _jiwer("wer", predictions, references) * 100,
        "cer": _jiwer("cer", predictions, references) * 100,
        "mer": _jiwer("mer", predictions, references) * 100,
        "wil": _jiwer("wil", predictions, references) * 100,
        "n_samples": len(references),
    }
    return metrics


def aggregate_by_domain(results: list[dict]) -> dict:
    """Compute metrics grouped by sample domain."""
    by_domain: dict[str, dict] = defaultdict(lambda: {"refs": [], "preds": []})
    for r in results:
        domain = r.get("domain", "other")
        by_domain[domain]["refs"].append(r["transcript"])
        by_domain[domain]["preds"].append(r.get("prediction", ""))
    return {
        domain: compute_metrics(data["preds"], data["refs"]) for domain, data in by_domain.items()
    }


def load_benchmark_subsets(samples: list[dict], domains: list[str]) -> list[dict]:
    """Keep samples belonging to any of the given domains."""
    return [s for s in samples if s.get("domain", "") in domains]


def evaluate_model(model_path: str, samples: list[dict], batch_size: int = 16) -> dict:
    """Transcribe, score globally, and break WER/CER down by domain."""
    logger.info("Evaluating %s on %d samples", model_path, len(samples))
    results = transcribe_batch(model_path, samples, batch_size=batch_size)
    refs = [r["transcript"] for r in results]
    preds = [r.get("prediction", "") for r in results]
    return {
        "overall": compute_metrics(preds, refs),
        "by_domain": aggregate_by_domain(results),
    }


def evaluate_and_compare(
    models: list[str], samples: list[dict], batch_size: int = 16
) -> dict[str, dict]:
    """Evaluate several models and format a comparison summary per model."""
    summaries: dict[str, dict] = {}
    for model_path in models:
        summary = evaluate_model(model_path, samples, batch_size=batch_size)
        summaries[model_path] = {
            "overall": summary["overall"],
            "by_domain": {
                domain: metrics["wer"] for domain, metrics in summary["by_domain"].items()
            },
        }
    return summaries
