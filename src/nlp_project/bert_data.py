"""Data layer for Q2: tokenization and split construction for BERT fine-tuning.

This module is *additive* on top of :mod:`nlp_project.data` — it reuses
``load_20ng`` and ``train_val_split`` (with ``seed=42``) so the validation
indices match Q1 by construction, which is what makes the Q1↔Q2 comparison
apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class TextClassificationDataset(torch.utils.data.Dataset):
    """A tiny ``torch.utils.data.Dataset`` over already-tokenized encodings.

    HuggingFace ``Trainer`` happily consumes a dataset that yields plain
    Python dicts containing ``input_ids``, ``attention_mask``, and
    ``labels`` — the default ``DataCollatorWithPadding`` then dynamically
    pads each batch. We keep this class minimal on purpose so unit tests
    can exercise it without a real tokenizer.
    """

    def __init__(
        self,
        encodings: dict[str, list[list[int]]],
        labels: Sequence[int],
    ) -> None:
        n = len(labels)
        for key, value in encodings.items():
            if len(value) != n:
                raise ValueError(
                    f"encodings[{key!r}] has length {len(value)}, expected {n}"
                )
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def tokenize(
    docs: Sequence[str],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Run a HuggingFace tokenizer over ``docs`` and return the dict shape
    that :class:`TextClassificationDataset` expects.

    Uses ``padding=False`` so the Trainer's collator can pad dynamically
    to the longest sequence in each batch — much cheaper than always
    padding to ``max_length``.
    """
    enc = tokenizer(
        list(docs),
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
    }


def build_splits(
    tokenizer_name: str,
    max_length: int,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """End-to-end: load 20NG, stratified train/val split, tokenize all three.

    Returns a dict with keys ``train``, ``val``, ``test`` (each a
    :class:`TextClassificationDataset`), ``label_names`` (length 20), and
    ``tokenizer`` (kept on the result so the caller can hand it to the
    Trainer as the ``processing_class``).
    """
    from transformers import AutoTokenizer

    from .data import load_20ng, train_val_split

    train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)
    tr_docs, tr_labels, val_docs, val_labels = train_val_split(
        train_docs, train_labels, val_frac=val_frac, seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return {
        "train": TextClassificationDataset(
            tokenize(tr_docs, tokenizer, max_length), tr_labels.tolist(),
        ),
        "val": TextClassificationDataset(
            tokenize(val_docs, tokenizer, max_length), val_labels.tolist(),
        ),
        "test": TextClassificationDataset(
            tokenize(test_docs, tokenizer, max_length), test_labels.tolist(),
        ),
        "label_names": label_names,
        "tokenizer": tokenizer,
    }
