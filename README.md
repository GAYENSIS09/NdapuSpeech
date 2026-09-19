# NdapuSpeech

**Open ASR for the Wolof language**

building a high-quality, open-source Automatic Speech Recognition model for Wolof, the most widely spoken language in Senegal. Named after Ndapu, my mother.

## Overview

NdapuSpeech aims to create a production-grade ASR model for Wolof by:

- Aggregating and cleaning all available public Wolof speech datasets
- Training on Whisper Large-v3 Turbo with LoRA fine-tuning with 152 hours off fine tuning on 16G RTX 5060
- Evaluating rigorously against baselines (M-Kiriku, Whisper zero-shot)

## Datasets Used

| Dataset                | Domain            | Hours | Source      |
| ---------------------- | ----------------- | ----- | ----------- |
| **Kallaama**     | Agriculture/Radio | ~55h  | OpenSLR 151 |
| **ALFFA**        | Read speech       | ~18h  | OpenSLR 25  |
| **FLEURS (wo)**  | Multi-domain      | ~10h  | Google/HF   |
| **WolBanking77** | Banking           | ~4h   | AI4D/HF     |

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



## Results (Target)

| Model                             | WER (dev)      | WER (test)     | Notes      |
| --------------------------------- | -------------- | -------------- | ---------- |
| Whisper Large-v3 (zero-shot)      | ~35%           | ~38%           | Baseline   |
| **NdapuSpeech (this work)** | **<15%** | **<18%** | Fine-tuned |

## Requirements

- Python 3.11+
- GPU (NVIDIA, 16GB+ VRAM recommended for training)
- `torch`, `transformers`, `datasets`, `accelerate`, `peft`

## Citation

```bibtex
@misc{ndapuspeech,
  title={NdapuSpeech: Open ASR for Wolof},
  author={Baye Mor Gaye},
  year={2024},
  url={https://github.com/GAYENSIS09/ndapuspeech}
}
```

## License

MIT License — see `LICENSE` for details.
