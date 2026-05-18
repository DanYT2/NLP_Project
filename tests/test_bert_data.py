"""Tests for src/nlp_project/bert_data.py."""

from __future__ import annotations

import pytest

from nlp_project import bert_data as bd


# ---------- fast unit tests: synthetic encodings, no tokenizer/network ----------


def test_dataset_len_matches_labels() -> None:
    encodings = {
        "input_ids": [[101, 7592, 102], [101, 2088, 102]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
    }
    ds = bd.TextClassificationDataset(encodings, labels=[3, 7])
    assert len(ds) == 2


def test_dataset_getitem_returns_all_keys() -> None:
    encodings = {
        "input_ids": [[101, 1, 102], [101, 2, 102], [101, 3, 102]],
        "attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    }
    ds = bd.TextClassificationDataset(encodings, labels=[0, 1, 19])
    item = ds[2]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"] == [101, 3, 102]
    assert item["labels"] == 19


def test_dataset_preserves_label_order() -> None:
    encodings = {"input_ids": [[i] for i in range(5)], "attention_mask": [[1] for _ in range(5)]}
    labels = [4, 0, 19, 7, 12]
    ds = bd.TextClassificationDataset(encodings, labels=labels)
    assert [ds[i]["labels"] for i in range(5)] == labels


# ---------- integration tests: real tokenizer; marked slow ----------


TEST_TOKENIZER = "bert-base-uncased"


@pytest.fixture(scope="module")
def bert_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEST_TOKENIZER)


@pytest.mark.slow
def test_tokenize_truncates_at_max_length(bert_tokenizer) -> None:
    long_doc = "word " * 1000
    enc = bd.tokenize([long_doc], bert_tokenizer, max_length=64)
    assert len(enc["input_ids"][0]) == 64
    assert len(enc["attention_mask"][0]) == 64


@pytest.mark.slow
def test_tokenize_attention_mask_marks_pad_free(bert_tokenizer) -> None:
    enc = bd.tokenize(["short doc"], bert_tokenizer, max_length=32)
    assert all(m == 1 for m in enc["attention_mask"][0])  # padding=False => no zero tokens


@pytest.mark.slow
def test_build_splits_has_three_splits_and_20_labels() -> None:
    splits = bd.build_splits(tokenizer_name=TEST_TOKENIZER, max_length=32)
    assert set(splits.keys()) >= {"train", "val", "test", "label_names"}
    assert len(splits["label_names"]) == 20
    # train+val should be the original 20NG train count; val is 10% of train+val.
    n_trainval = len(splits["train"]) + len(splits["val"])
    assert abs(len(splits["val"]) / n_trainval - 0.10) < 0.01


@pytest.mark.slow
def test_build_splits_matches_q1_val_indices() -> None:
    """Q1↔Q2 parity: same seed/val_frac on the same raw train docs must produce
    a val split with the same labels in the same order as data.train_val_split."""
    from nlp_project.data import load_20ng, train_val_split

    train_docs, train_labels, _, _, _ = load_20ng(remove=True)
    _, _, _, q1_val_labels = train_val_split(train_docs, train_labels, val_frac=0.1, seed=42)
    splits = bd.build_splits(tokenizer_name=TEST_TOKENIZER, max_length=32)
    q2_val_labels = [splits["val"][i]["labels"] for i in range(len(splits["val"]))]
    assert list(q2_val_labels) == list(q1_val_labels)
