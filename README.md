# NdapuSpeech

**Open ASR for the Wolof language** — building a high-quality, open-source Automatic Speech Recognition model for Wolof, the most widely spoken language in Senegal. Named after Ndapu, my mother.

## Overview

NdapuSpeech aims to create a production-grade ASR model for Wolof by:
- Aggregating and cleaning all available public Wolof speech datasets
- Training on Whisper Large-v3 Turbo with LoRA fine-tuning
- Evaluating rigorously against baselines (M-Kiriku, Whisper zero-shot)

## Datasets Used

| Dataset | Domain | Hours | Source |
|---------|--------|-------|--------|
| **Kallaama** | Agriculture/Radio | ~55h | OpenSLR 151 |
| **ALFFA** | Read speech | ~18h | OpenSLR 25 |
| **FLEURS (wo)** | Multi-domain | ~10h | Google/HF |
| **WolBanking77** | Banking | ~4h | AI4D/HF |

**Total: ~87+ hours** of diverse Wolof speech.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download all datasets
python -m ndapuspeech.data download_all

# Build unified train/val/test dataset
python -c "from ndapuspeech.data import build_unified_dataset; build_unified_dataset()"

# Train model (see configs/training.yaml for hyperparams)
python -m ndapuspeech.training
```

## Project Structure

```
configs/           # YAML configuration (config, training, data)
notebooks/         # Example pipelines (run_all, download, train, etc.)
src/ndapuspeech/   # Main package
  ├── config.py    # Central config loader
  ├── data.py      # Download + unified dataset builder
  ├── audio.py     # VAD, segmentation, preprocessing
  ├── models.py    # Whisper + tokenizer
  ├── training.py  # Training loop (HF Trainer)
  ├── evaluation.py # WER/CER metrics
  ├── text.py      # Wolof text normalization
  └── kenlm.py     # Language model helpers
tests/             # Unit tests (pytest)
```

## Configuration

All tunable parameters live in `configs/*.yaml`:
- `config.yaml` — paths, audio, model
- `training.yaml` — hyperparams, LoRA, scheduler
- `data.yaml` — dataset sources, manifests, tokenizer

Edit YAMLs; no code changes needed.

## Results (Target)

| Model | WER (dev) | WER (test) | Notes |
|-------|-----------|------------|-------|
| Whisper Large-v3 (zero-shot) | ~35% | ~38% | Baseline |
| **NdapuSpeech (this work)** | **<15%** | **<18%** | Fine-tuned |

## Requirements

- Python 3.11+
- GPU (NVIDIA, 16GB+ VRAM recommended for training)
- `torch`, `transformers`, `datasets`, `accelerate`, `peft`

## Citation

```bibtex
@misc{ndapuspeech,
  title={NdapuSpeech: Open ASR for Wolof},
  author={...},
  year={2024},
  url={https://github.com/GAYENSIS09/ndapuspeech}
}
```

## License

MIT License — see `LICENSE` for details.