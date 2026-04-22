"""Training loop. Callable from the Colab notebook.

Writes per-run metrics and raw test predictions to disk — everything a later
analysis pass needs to re-derive CIs without retraining.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pv_ade.data import apply_splits, load_ade_corpus, load_splits
from pv_ade.evaluate import classification_metrics
from pv_ade.model import build_model


@dataclass
class TrainConfig:
    """Hyperparameters for a single training run.

    Defaults are representative placeholders — the ablation always passes
    values through from ``configs/ablation.yaml`` via ``train_one_run``'s
    ``config`` dict, so changing the defaults here does not change the
    ablation's behavior. ``learning_rate`` and ``num_epochs`` must be set
    by the caller; ``train_one_run`` raises if either is left unset.
    """

    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 3e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and HF for reproducibility."""
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _train_config_from_dict(config: dict[str, Any]) -> TrainConfig:
    """Project a raw config dict down to the fields ``TrainConfig`` knows about."""
    fields = TrainConfig.__dataclass_fields__
    return TrainConfig(**{k: v for k, v in config.items() if k in fields})


def train_one_run(
    model_name: str,
    checkpoint: str,
    seed: int,
    config: dict[str, Any],
    splits_path: Path,
    metrics_dir: Path,
    predictions_dir: Path,
    eval_split: str = "test",
) -> dict[str, Any]:
    """Train one (model, seed) run.

    Loads ADE Corpus v2, applies the committed split, fine-tunes from `checkpoint`,
    evaluates on `eval_split` (default "test"), and writes both metrics JSON and
    raw predictions to disk.
    """
    import torch
    from transformers import (
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    if config.get("learning_rate") is None or config.get("num_epochs") is None:
        raise ValueError(
            "learning_rate and num_epochs must be set in config — pick them from the sweep first"
        )
    train_cfg = _train_config_from_dict(config)
    _seed_everything(seed)

    raw = load_ade_corpus()
    splits = load_splits(splits_path)
    parts = apply_splits(raw, splits)

    model, tokenizer = build_model(checkpoint)

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=train_cfg.max_length,
        )

    encoded = {name: ds.map(tokenize, batched=True) for name, ds in parts.items()}
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{model_name}_seed{seed}"

    args = TrainingArguments(
        output_dir=str(metrics_dir / "_hf_tmp" / run_id),
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size * 2,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["val"],
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    eval_ds = encoded[eval_split]
    pred_output = trainer.predict(eval_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    y_pred = probs.argmax(axis=-1)
    y_prob_pos = probs[:, 1]
    y_true = np.asarray(eval_ds["label"])

    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob_pos)

    record = {
        "model": model_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "config": asdict(train_cfg),
        "eval_split": eval_split,
        "metrics": metrics,
    }
    (metrics_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2))

    np.savez_compressed(
        predictions_dir / f"{run_id}.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_prob_pos=y_prob_pos,
    )

    return record
