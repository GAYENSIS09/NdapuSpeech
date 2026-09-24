"""Orchestrated fine-tuning loop for Whisper-based NdapuSpeech ASR."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from . import config
from .evaluation import compute_cer, compute_wer
from .models import (
    DataCollatorSpeechSeq2SeqWithPadding,
    NoisyDataset,
    build_model,
    set_seed,
)

logger = logging.getLogger(__name__)

_JSON_SPLITS = ("train", "val", "test")

_YAML_REMAP = {
    "num_epochs": "max_epochs",
    "grad_accum": "gradient_accumulation",
    "lr_decoder": "learning_rate_decoder",
    "lr_encoder": "learning_rate_encoder",
    "patience": "early_stop_patience",
}


@dataclass
class TrainingConfig:
    """All knobs for the training loop, defaults sourced from `configs/training.yaml`."""

    data_dir: Path = field(
        default_factory=lambda: config.TRAINING_DATA_DIR or Path("data/processed/ndapuspeech_ds")
    )
    output_dir: Path = field(
        default_factory=lambda: config.TRAINING_OUTPUT_DIR or Path("models/ndapuspeech-v1")
    )
    base_model: str = config.BASE_MODEL
    tokenizer_dir: str | None = field(
        default_factory=lambda: (
            str(config.EXTENDED_TOKENIZER_DIR) if config.EXTENDED_TOKENIZER_DIR.exists() else None
        )
    )
    num_epochs: int = config.MAX_EPOCHS
    batch_size: int = config.BATCH_SIZE
    grad_accum: int = config.GRADIENT_ACCUMULATION
    lr_decoder: float = config.LEARNING_RATE_DECODER
    lr_encoder: float = config.LEARNING_RATE_ENCODER
    warmup_steps: int = config.WARMUP_STEPS
    patience: int = config.EARLY_STOP_PATIENCE
    noise_dir: Path | None = field(default_factory=lambda: config.TRAINING_NOISE_DIR)
    use_lora: bool = config.TRAINING_USE_LORA
    resume_from: str | None = field(
        default_factory=lambda: (
            str(config.TRAINING_RESUME_FROM) if config.TRAINING_RESUME_FROM else None
        )
    )
    max_steps: int | None = config.TRAINING_MAX_STEPS
    eval_steps: int = config.TRAINING_EVAL_STEPS
    save_steps: int = config.TRAINING_SAVE_STEPS
    seed: int = config.SEED
    push_to_hub: str | None = config.TRAINING_PUSH_TO_HUB
    fp16: bool = config.TRAINING_FP16
    bf16: bool = config.TRAINING_BF16
    hub_dataset: str = config.HF_DATASET_REPO

    @classmethod
    def from_yaml(cls, overrides: dict | None = None) -> "TrainingConfig":
        """Build a TrainingConfig from `configs/training.yaml`, with optional overrides."""
        values: dict = {}
        for field_name in cls.__dataclass_fields__:
            yaml_key = _YAML_REMAP.get(field_name, field_name)
            if yaml_key in config.TRAINING:
                values[field_name] = config.TRAINING[yaml_key]
        for key in ("data_dir", "output_dir"):
            if values.get(key) is not None:
                values[key] = Path(str(values[key]))
        for key in ("resume_from", "noise_dir"):
            if values.get(key) is not None:
                values[key] = str(values[key])
        if overrides:
            values.update(overrides)
        cfg = cls(**values)
        if cfg.save_steps % cfg.eval_steps != 0:
            raise ValueError(
                f"save_steps ({cfg.save_steps}) must be a multiple of eval_steps "
                f"({cfg.eval_steps}) when load_best_model_at_end=True"
            )
        return cfg


def _load_training_libs() -> tuple[Any, ...]:
    """Import heavy training dependencies once, raising a typed error if missing."""
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperProcessor,
        )

        class _AutocastSeq2SeqTrainer(Seq2SeqTrainer):
            """Seq2SeqTrainer that wraps generate() in autocast to fix fp16 dtype mismatches."""

            def prediction_step(
                self, model, inputs, prediction_loss_only=None, ignore_keys=None, **gen_kwargs
            ):
                import contextlib

                if not getattr(self.args, "predict_with_generate", False) or prediction_loss_only:
                    return super().prediction_step(
                        model,
                        inputs,
                        prediction_loss_only=prediction_loss_only,
                        ignore_keys=ignore_keys,
                        **gen_kwargs,
                    )

                if not hasattr(gen_kwargs, "max_new_tokens") and "max_new_tokens" not in gen_kwargs:
                    gen_kwargs.setdefault("max_new_tokens", 225)

                device = self.args.device
                dtype = (
                    torch.float16
                    if self.args.fp16
                    else (torch.bfloat16 if self.args.bf16 else torch.float32)
                )
                autocast_ctx = (
                    torch.amp.autocast(device_type=device.type, dtype=dtype)
                    if dtype != torch.float32
                    else contextlib.nullcontext()
                )
                with autocast_ctx, torch.no_grad():
                    # The Trainer stubs type ``model`` as optional; at this
                    # point prediction_step can only run with a model set.
                    model = cast(Any, self.model)
                    generated_tokens = model.generate(
                        inputs["input_features"],
                        attention_mask=inputs.get("attention_mask"),
                        **gen_kwargs,
                    )

                labels = inputs.get("labels")
                if labels is not None:
                    labels = labels.to(generated_tokens.device)

                loss = None

                return loss, generated_tokens, labels

        return (
            torch,
            load_dataset,
            WhisperProcessor,
            _AutocastSeq2SeqTrainer,
            Seq2SeqTrainingArguments,
            EarlyStoppingCallback,
        )
    except ImportError as e:
        raise RuntimeError("Missing dependencies: pip install -r requirements.txt") from e


def compute_metrics_fn(processor: Any) -> Callable[[Any], dict]:
    """Return a trainer metrics callback decoding predictions to WER/CER."""

    def compute_metrics(pred: Any) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": compute_wer(pred_str, label_str) * 100,
            "cer": compute_cer(pred_str, label_str) * 100,
        }

    return compute_metrics


def _load_datasets(cfg: TrainingConfig) -> Any:
    """Load train/val/test splits from local JSONL or a hub dataset."""
    _, load_dataset, _, _, _, _ = _load_training_libs()

    if cfg.data_dir.exists():
        data_files = {split: str(cfg.data_dir / f"{split}.jsonl") for split in _JSON_SPLITS}
        ds = load_dataset("json", data_files=data_files)
    else:
        logger.info("Data dir %s not found; trying hub dataset...", cfg.data_dir)
        try:
            ds = load_dataset(cfg.hub_dataset, trust_remote_code=True)
        except (ConnectionError, OSError, ValueError) as e:
            raise FileNotFoundError("No local or hub dataset found") from e
    return ds


def train_model(cfg: TrainingConfig) -> Path:
    """Run the full training pipeline and return the saved model directory."""
    torch, _, processor_cls, trainer_cls, training_args_cls, early_stopping_cls = (
        _load_training_libs()
    )
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    processor = processor_cls.from_pretrained(cfg.base_model)
    if cfg.tokenizer_dir:
        from transformers import AutoTokenizer

        processor.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
        logger.info("Loaded extended tokenizer from %s", cfg.tokenizer_dir)

    model, param_groups = build_model(
        cfg.base_model,
        use_lora=cfg.use_lora,
        resume_from=cfg.resume_from,
        lr_decoder=cfg.lr_decoder,
        lr_encoder=cfg.lr_encoder,
        tokenizer_path=cfg.tokenizer_dir,
    )
    processor.tokenizer.model_max_length = model.config.max_target_positions

    ds = _load_datasets(cfg)
    train_ds = ds["train"].select_columns(["audio", "transcript"])
    val_ds = (ds["val"] if "val" in ds else ds["test"]).select_columns(["audio", "transcript"])
    test_ds = ds.get("test", None)

    del ds

    if cfg.noise_dir:
        train_ds = NoisyDataset(train_ds, processor, noise_dir=cfg.noise_dir, aug_prob=0.5)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_steps % cfg.eval_steps != 0:
        raise ValueError(
            f"save_steps ({cfg.save_steps}) must be a multiple of eval_steps "
            f"({cfg.eval_steps}) when load_best_model_at_end=True"
        )

    optim_args = {
        "num_train_epochs": cfg.num_epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.lr_decoder,
        "warmup_steps": cfg.warmup_steps,
        "eval_strategy": "steps",
        "eval_steps": cfg.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "predict_with_generate": True,
        "fp16": cfg.fp16 or (not cfg.bf16 and device == "cuda"),
        "bf16": cfg.bf16,
        "logging_steps": 25,
        "output_dir": str(out_dir),
        "report_to": "none",
        "seed": cfg.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "save_total_limit": 3,
    }
    if cfg.max_steps:
        optim_args["max_steps"] = cfg.max_steps

    training_args = training_args_cls(**optim_args)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(processor),
        callbacks=[early_stopping_cls(early_stopping_patience=cfg.patience)],
    )
    if not cfg.use_lora:
        trainer.optimizer = torch.optim.AdamW(param_groups)

    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))
    trainer.train()

    logger.info("Saving final model...")
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    if test_ds is not None:
        logger.info("Test: %s", trainer.evaluate(test_ds, metric_key_prefix="test"))

    if cfg.push_to_hub:
        model.push_to_hub(cfg.push_to_hub)
        processor.push_to_hub(cfg.push_to_hub)

    logger.info("Model saved to %s", out_dir)
    return out_dir
