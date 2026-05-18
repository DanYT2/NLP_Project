"""MLM pretraining orchestration for Q3 — continue BERT's masked-LM objective
on the 20 Newsgroups training corpus.

Q3's hypothesis: doing one extra stage of in-domain MLM pretraining before
classification fine-tuning produces a better encoder than fine-tuning
``bert-base-uncased`` directly (the Q2b baseline).

This module is the Stage-A counterpart to :mod:`nlp_project.bert_train`'s
Stage-B fine-tuning. Stage B itself is unchanged — Q3's classification
notebook reuses :func:`bert_train.run_finetune` verbatim, but hands it the
MLM-pretrained encoder instead of letting it download fresh
``bert-base-uncased`` weights.

Design notes:

* The classification ``val`` set carved by :func:`data.train_val_split` is
  **not** used as the MLM val set — that would leak the classification
  val into the MLM stage and contaminate the Stage-B early stopping. We
  carve an independent 90/10 split out of the train docs *inside*
  :func:`run_mlm_pretrain` purely so the Trainer can compute
  ``eval_loss`` for early stopping.
* ``DataCollatorForLanguageModeling`` does the 80/10/10 mask/random/keep
  token logic for us — re-implementing it adds no educational value vs.
  Q2's HF-Trainer story.
* Perplexity is reported alongside loss because it is the conventional
  language-modelling metric (``perplexity = exp(loss)``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from . import SEED


class _MlmDataset(torch.utils.data.Dataset):
    """Minimal torch Dataset over pre-tokenized encodings (no labels).

    HuggingFace's ``DataCollatorForLanguageModeling`` produces the
    ``labels`` tensor on the fly by masking ``input_ids``, so the dataset
    must NOT carry a ``labels`` field — keeping one here would shadow
    the collator's masked labels and break MLM training.
    """

    def __init__(self, encodings: dict[str, list[list[int]]]) -> None:
        lengths = {len(v) for v in encodings.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"encodings have inconsistent lengths: {lengths!r}",
            )
        self.encodings = encodings
        self._n = next(iter(lengths))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return {k: v[idx] for k, v in self.encodings.items()}


def build_mlm_dataset(
    tokenizer: Any,
    docs: Sequence[str],
    max_length: int,
) -> _MlmDataset:
    """Tokenize ``docs`` and wrap them in a torch Dataset suitable for the
    HuggingFace ``DataCollatorForLanguageModeling`` to mask on the fly.

    Uses ``padding=False`` so the collator pads dynamically per batch.
    """
    enc = tokenizer(
        list(docs),
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return _MlmDataset({
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
    })


def make_mlm_training_args(
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
    """Build a ``TrainingArguments`` for MLM pretraining.

    Differences vs :func:`bert_train.make_training_args`:
    ``metric_for_best_model="eval_loss"``, ``greater_is_better=False``
    (lower MLM loss is better; no accuracy/F1 here).
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=logging_steps,
        report_to=list(report_to) if report_to is not None else ["wandb"],
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )


def _split_for_mlm_eval(
    docs: Sequence[str], val_frac: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Random 90/10 split of ``docs`` for MLM train/eval.

    No stratification — there are no labels at this stage. Uses
    sklearn's ``train_test_split`` for determinism via ``random_state``.
    """
    from sklearn.model_selection import train_test_split

    train_d, eval_d = train_test_split(
        list(docs), test_size=val_frac, random_state=seed, shuffle=True,
    )
    return train_d, eval_d


def run_mlm_pretrain(
    *,
    tokenizer: Any,
    train_docs: Sequence[str],
    out_dir: Path | str,
    model: Any | None = None,
    model_name: str | None = None,
    lr: float = 5e-5,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    mlm_probability: float = 0.15,
    mlm_val_frac: float = 0.1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    run_name: str = "q3-mlm-pretrain",
    report_to: Sequence[str] | None = None,
    logging_steps: int = 50,
) -> dict[str, Any]:
    """Run domain-adaptive MLM pretraining and save the resulting checkpoint.

    Exactly one of ``model`` (pre-instantiated, for tests) or ``model_name``
    (a HuggingFace Hub ID such as ``"bert-base-uncased"``) must be given.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer matching ``model``/``model_name``.
    train_docs:
        Raw documents to pretrain on. We carve an internal 90/10 train/eval
        split out of these so the Trainer can compute ``eval_loss`` for
        early stopping; the eval split is independent of the
        classification-stage validation set.
    out_dir:
        Where to write the MLM checkpoint (``trainer.save_model``).

    Returns
    -------
    dict with keys:
        ``out_dir`` (Path), ``log_history`` (list of dicts from
        ``trainer.state.log_history``), ``best_eval_loss`` (float), and
        ``perplexity`` (``exp(best_eval_loss)``).
    """
    from transformers import (
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
    )

    if (model is None) == (model_name is None):
        raise ValueError("provide exactly one of `model` or `model_name`")
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_docs, eval_docs = _split_for_mlm_eval(
        train_docs, val_frac=mlm_val_frac, seed=SEED,
    )
    train_ds = build_mlm_dataset(tokenizer, tr_docs, max_length=max_length)
    eval_ds = build_mlm_dataset(tokenizer, eval_docs, max_length=max_length)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )

    args = make_mlm_training_args(
        out_dir=out_dir,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        run_name=run_name,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        report_to=report_to,
        logging_steps=logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    best_eval_loss = float(trainer.state.best_metric) if trainer.state.best_metric is not None else float(trainer.state.log_history[-1].get("eval_loss", float("nan")))
    perplexity = math.exp(best_eval_loss) if math.isfinite(best_eval_loss) else float("nan")

    return {
        "out_dir": out_dir,
        "log_history": list(trainer.state.log_history),
        "best_eval_loss": best_eval_loss,
        "perplexity": perplexity,
    }
