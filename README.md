<div align="center">
  <img src="ndapu.svg" alt="Ndapu logo" width="200" />
  <h1>NdapuSpeech</h1>
</div>



*An open-source ASR model for the Wolof language.*

<!-- ![Ndapu logo](assets/logo.png) -->

NdapuSpeech is an open Automatic Speech Recognition (ASR) model for **Wolof**, the most widely spoken language in Senegal. The project is named after **Ndapu**, my mother.

## Overview

NdapuSpeech aims to build a production-grade ASR model for Wolof by:

- Aggregating and cleaning all available public Wolof speech datasets
- Fine-tuning **Whisper Large-v3 Turbo** with **LoRA**
- Evaluating rigorously against baselines (M-Kiriku, Whisper zero-shot)

**Training setup:** 152 hours of fine-tuning on a single NVIDIA RTX 5060 (16GB VRAM).

## Demo

<!-- [Watch the demo video](INSERT_LINK) -->

*Demo video coming soon.*

## Resources

| Resource      | Link                                                                          |
| ------------- | ----------------------------------------------------------------------------- |
| Model weights | [ndapuspeech-asr-v1](INSERT_HUGGINGFACE_LINK) on Hugging Face                  |
| Code          | [github.com/GAYENSIS09/ndapuspeech](https://github.com/GAYENSIS09/ndapuspeech) |

## Datasets Used

| Dataset               | Domain              | Hours | Source      |
| --------------------- | ------------------- | ----- | ----------- |
| **Kallaama**    | Agriculture / Radio | ~55h  | OpenSLR 151 |
| **ALFFA**       | Read speech         | ~18h  | OpenSLR 25  |
| **FLEURS (wo)** | Multi-domain        | ~10h  | Google / HF |

**Total: ~80.5+ hours** of diverse Wolof speech.

## Results (Target)

> These are the project's target metrics, not yet measured final results.

| Model                             | WER (dev)      | WER (test)     | Notes      |
| --------------------------------- | -------------- | -------------- | ---------- |
| Whisper Large-v3 (zero-shot)      | ~35%           | ~38%           | Baseline   |
| **NdapuSpeech (this work)** | **<15%** | **<18%** | Fine-tuned |

## Requirements

- Python 3.11+
- NVIDIA GPU, 16GB+ VRAM recommended for training
- `torch`, `transformers`, `datasets`, `accelerate`, `peft`

## Installation

```bash
git clone https://github.com/GAYENSIS09/ndapuspeech.git
cd ndapuspeech
pip install -r requirements.txt
```

## Citation

```bibtex
@misc{ndapuspeech,
  title={NdapuSpeech: Open ASR for Wolof},
  author={Baye Mor Gaye},
  year={2026},
  url={https://github.com/GAYENSIS09/ndapuspeech}
}
```

## License

MIT License — see `LICENSE` for details.
