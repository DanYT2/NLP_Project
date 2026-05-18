"""Fine-tuning orchestration for Q2 — BERT on 20 Newsgroups via HF Trainer.

The HF Trainer handles a lot of plumbing for us (LR schedule, eval loop,
W&B reporting, checkpointing, restore-best-on-end). What this module adds
is project-specific glue:

* MPS-safe defaults (no fp16/bf16, no pinned memory).
* The seed wiring from :data:`nlp_project.SEED`.
* A ``compute_metrics`` that matches :func:`nlp_project.eval.metrics_from_predictions`.
* An optional encoder freeze for the linear-probe variant.
* A ``run_finetune`` that returns a single dict with both the
  best validation metric and the full test-set metrics dict.

Note on W&B: we use ``report_to=["wandb"]`` from Trainer. The caller is
responsible for ``wandb.init(project=..., name=..., config=...)`` before
:func:`run_finetune` and ``wandb.finish()`` after — that mirrors the Q1
notebook pattern and keeps the training function pure with respect to
W&B session management.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import SEED
from .eval import metrics_from_predictions


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """HF Trainer's ``compute_metrics`` hook.

    Accepts either a transformers ``EvalPrediction`` (which exposes
    ``.predictions`` and ``.label_ids``) or a plain ``(logits, labels)``
    tuple — both shapes appear depending on Trainer version.
    """
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(
            __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                labels, preds, average="macro", zero_division=0,
            ),
        ),
    }


def freeze_encoder(model: Any) -> None:
    """Set ``requires_grad=False`` on every parameter except the
    classification head — used for the linear-probe ablation.

    Works for ``BertForSequenceClassification`` and its siblings: the
    head module is always exposed at ``model.classifier``.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def make_training_args(
    out_dir: Path | str,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    run_name: str,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    report_to: Sequence[str] | None = None,
    logging_steps: int = 50,
) -> Any:
    """Build a ``TrainingArguments`` with MPS-safe defaults.

    Caller passes ``report_to=["wandb"]`` from notebooks and
    ``report_to=[]`` from tests to keep tests offline.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        report_to=list(report_to) if report_to is not None else ["wandb"],
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )


def _make_model(
    model_name: str,
    label_names: list[str],
    freeze_encoder_: bool,
) -> Any:
    from transformers import AutoModelForSequenceClassification

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    if freeze_encoder_:
        freeze_encoder(model)
    return model


def run_finetune(
    splits: dict[str, Any],
    *,
    out_dir: Path | str,
    lr: float,
    epochs: int,
    batch_size: int,
    run_name: str,
    model_name: str | None = None,
    model: Any | None = None,
    freeze_encoder_: bool = False,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    report_to: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run an end-to-end fine-tuning experiment and return its metrics.

    Exactly one of ``model_name`` (Hub ID, e.g. ``"bert-base-uncased"``)
    or ``model`` (pre-instantiated for tests with a tiny config) must be
    provided.

    Returns
    -------
    A dict ``{"test_metrics": <metrics_from_predictions output>,
    "best_eval_metric": float, "trainer": Trainer}``. The trainer is
    returned so the notebook can inspect ``log_history`` for plots.
    """
    from transformers import DataCollatorWithPadding, Trainer

    if (model is None) == (model_name is None):
        raise ValueError("provide exactly one of `model_name` or `model`")
    if model is None:
        model = _make_model(model_name, splits["label_names"], freeze_encoder_)
    elif freeze_encoder_:
        freeze_encoder(model)

    tokenizer = splits["tokenizer"]
    args = make_training_args(
        out_dir=out_dir,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        run_name=run_name,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    best_eval = trainer.state.best_metric

    test_pred = trainer.predict(splits["test"])
    logits = test_pred.predictions
    labels = test_pred.label_ids
    y_pred = np.argmax(logits, axis=-1)
    test_metrics = metrics_from_predictions(labels, y_pred, splits["label_names"])

    return {
        "test_metrics": test_metrics,
        "best_eval_metric": best_eval,
        "trainer": trainer,
    }
