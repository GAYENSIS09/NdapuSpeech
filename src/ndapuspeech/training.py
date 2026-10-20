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
    merge_lora_weights,
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
    tokenizer_dir: str | None = None
    extended_tokenizer: bool = config.TRAINING_EXTENDED_TOKENIZER
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
    preprocessed_dir: Path | None = field(
        default_factory=lambda: config.TRAINING_PREPROCESSED_DIR
    )
    group_by_length: bool = config.TRAINING_GROUP_BY_LENGTH
    dataloader_num_workers: int = config.TRAINING_DATALOADER_NUM_WORKERS
    dataloader_pin_memory: bool = config.TRAINING_DATALOADER_PIN_MEMORY
    attn_implementation: str | None = config.TRAINING_ATTN_IMPLEMENTATION
    load_in_4bit: bool = config.TRAINING_LOAD_IN_4BIT
    torch_compile: bool = config.TRAINING_TORCH_COMPILE
    num_proc: int = config.TRAINING_NUM_PROC
    eval_max_samples: int = config.TRAINING_EVAL_MAX_SAMPLES

    @classmethod
    def from_yaml(cls, overrides: dict | None = None) -> "TrainingConfig":
        """Build a TrainingConfig from `configs/training.yaml`, with optional overrides."""
        values: dict = {}
        for field_name in cls.__dataclass_fields__:
            yaml_key = _YAML_REMAP.get(field_name, field_name)
            if yaml_key in config.TRAINING:
                values[field_name] = config.TRAINING[yaml_key]
        for key in ("data_dir", "output_dir", "preprocessed_dir"):
            if values.get(key) is not None:
                values[key] = Path(str(values[key]))
        for key in ("resume_from", "noise_dir", "tokenizer_dir"):
            if values.get(key) is not None:
                values[key] = str(values[key])
        if overrides:
            values.update(overrides)
        cfg = cls(**values)
        if cfg.extended_tokenizer and not cfg.tokenizer_dir:
            cfg.tokenizer_dir = (
                str(config.EXTENDED_TOKENIZER_DIR) if config.EXTENDED_TOKENIZER_DIR.exists() else None
            )
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
    """Load train/val/test splits from preprocessed Arrow, JSONL, or hub."""
    _, load_dataset, _, _, _, _ = _load_training_libs()

    if cfg.preprocessed_dir is not None and cfg.preprocessed_dir.exists():
        from .prepare import is_preprocessed, load_preprocessed_dataset

        if is_preprocessed(cfg.preprocessed_dir):
            logger.info("Using preprocessed Arrow at %s (mmap, 0x I/O)", cfg.preprocessed_dir)
            return load_preprocessed_dataset(cfg.preprocessed_dir)

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


def _select_split_columns(split: Any) -> Any:
    """Keep precomputed columns when present, else raw audio + transcript."""
    names: list[str] = list(split.column_names)
    if "input_features" in names and "labels" in names:
        keep = ["input_features", "labels"]
        for extra in ("length", "duration"):
            if extra in names:
                keep.append(extra)
        return split.select_columns(keep)
    return split.select_columns(["audio", "transcript"])


def _length_grouping_kwargs(
    training_args_cls: Any, train_ds: Any, group_by_length: bool
) -> dict[str, Any]:
    """Map the note.txt length-grouping knob onto the installed transformers API.

    transformers 4.x takes ``group_by_length``; 5.x takes
    ``train_sampling_strategy="group_by_length"`` plus ``length_column_name``.
    """
    import inspect

    params = inspect.signature(training_args_cls.__init__).parameters
    if "group_by_length" in params:
        return {"group_by_length": group_by_length}
    if not group_by_length:
        return {"train_sampling_strategy": "random"}
    names = list(train_ds.column_names) if hasattr(train_ds, "column_names") else []
    if "length" in names:
        length_col = "length"
    elif "duration" in names:
        length_col = "duration"
    else:
        logger.warning("No length/duration column; length grouping falls back to random")
        return {"train_sampling_strategy": "random"}
    return {"train_sampling_strategy": "group_by_length", "length_column_name": length_col}


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
        attn_implementation=cfg.attn_implementation,
        load_in_4bit=cfg.load_in_4bit,
        torch_compile=cfg.torch_compile,
    )
    processor.tokenizer.model_max_length = model.config.max_target_positions

    ds = _load_datasets(cfg)
    train_ds = _select_split_columns(ds["train"])
    val_ds = _select_split_columns(ds["val"] if "val" in ds else ds["test"])
    if cfg.eval_max_samples and len(val_ds) > cfg.eval_max_samples:
        step = len(val_ds) / cfg.eval_max_samples
        idx = [int(i * step) for i in range(cfg.eval_max_samples)]
        val_ds = val_ds.select(idx)
        logger.info("Eval capped to %d evenly-spaced samples for speed", cfg.eval_max_samples)
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

    import os

    num_workers = cfg.dataloader_num_workers
    if os.name == "nt" and num_workers > 0:
        logger.warning(
            "[WIN] dataloader_num_workers=%d forced to 0 (Windows spawn unsafe)",
            num_workers,
        )
        num_workers = 0

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
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": cfg.dataloader_pin_memory and device == "cuda",
        **_length_grouping_kwargs(training_args_cls, train_ds, cfg.group_by_length),
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

    if cfg.use_lora:
        model = merge_lora_weights(model)
        trainer.model = model

    logger.info("Saving final model...")
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    if test_ds is not None:
        try:
            subset = test_ds
            try:
                if len(test_ds) > 100:
                    subset = test_ds.select(range(100))
            except (AttributeError, IndexError, ValueError):
                pass
            logger.info("Test: %s", trainer.evaluate(subset, metric_key_prefix="test"))
        except (RuntimeError, MemoryError) as e:
            logger.warning("[SKIP] final test eval failed (%s); model still saved", e)

    if cfg.push_to_hub:
        model.push_to_hub(cfg.push_to_hub)
        processor.push_to_hub(cfg.push_to_hub)

    logger.info("Model saved to %s", out_dir)
    try:
        import json

        (out_dir / "training_complete.json").write_text(
            json.dumps(
                {
                    "global_step": int(getattr(trainer.state, "global_step", 0)),
                    "base_model": cfg.base_model,
                    "max_epochs": cfg.num_epochs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Could not write training_complete.json (%s)", e)
    return out_dir
