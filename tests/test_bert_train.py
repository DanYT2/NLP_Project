"""Tests for src/nlp_project/bert_train.py.

The end-to-end smoke test deliberately builds a *tiny* BertForSequenceClassification
from a fresh BertConfig so the test never has to download multi-hundred-MB
model weights from the Hub.
"""

from __future__ import annotations

import pytest

from nlp_project import bert_data as bd
from nlp_project import bert_train as bt


# ---------- fast unit tests ----------


def test_make_training_args_propagates_hparams(tmp_path) -> None:
    args = bt.make_training_args(
        out_dir=tmp_path,
        lr=3e-5,
        epochs=2,
        batch_size=8,
        run_name="unit-test-run",
    )
    assert args.learning_rate == 3e-5
    assert args.num_train_epochs == 2
    assert args.per_device_train_batch_size == 8
    assert args.per_device_eval_batch_size == 8
    assert args.run_name == "unit-test-run"


def test_make_training_args_has_mps_safe_flags(tmp_path) -> None:
    """fp16/bf16 must be off on Apple Silicon; pin_memory must be off."""
    args = bt.make_training_args(out_dir=tmp_path, lr=2e-5, epochs=1, batch_size=4, run_name="x")
    assert args.fp16 is False
    assert args.bf16 is False
    assert args.dataloader_pin_memory is False


def test_make_training_args_load_best_on_macro_f1(tmp_path) -> None:
    args = bt.make_training_args(out_dir=tmp_path, lr=2e-5, epochs=1, batch_size=4, run_name="x")
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_macro_f1"
    assert args.greater_is_better is True


def test_compute_metrics_perfect() -> None:
    import numpy as np

    class _EvalPred:
        # HF Trainer passes either EvalPrediction or a (logits, labels) tuple.
        predictions = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        label_ids = np.array([0, 1, 1])

    metrics = bt.compute_metrics(_EvalPred())
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compute_metrics_accepts_tuple() -> None:
    import numpy as np

    logits = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    metrics = bt.compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0


def test_freeze_encoder_only_classifier_trains() -> None:
    """After freeze_encoder, only the classification head should have grads."""
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=128, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        num_labels=3,
    )
    model = BertForSequenceClassification(cfg)
    bt.freeze_encoder(model)

    encoder_grads = [p.requires_grad for n, p in model.named_parameters() if "classifier" not in n]
    head_grads = [p.requires_grad for n, p in model.named_parameters() if "classifier" in n]
    assert not any(encoder_grads), "encoder params should be frozen"
    assert all(head_grads), "classifier head should remain trainable"
    assert len(head_grads) >= 2  # weight + bias


# ---------- slow integration smoke test ----------


@pytest.mark.slow
def test_run_finetune_smoke(tmp_path) -> None:
    """End-to-end: tiny BertForSequenceClassification trained 1 epoch on a toy set.

    Verifies the Trainer wires up cleanly and run_finetune returns the
    expected dict shape. We don't make claims about accuracy here — the
    model is randomly initialised and the data is tiny.
    """
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Build a tiny BERT (no Hub download for weights) reusing the real vocab.
    cfg = BertConfig(
        vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        num_labels=3,
    )
    model = BertForSequenceClassification(cfg)

    # Toy data: 12 docs across 3 classes.
    docs = ["sports football game"] * 4 + ["politics election vote"] * 4 + ["computer software code"] * 4
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    enc = bd.tokenize(docs, tok, max_length=16)
    ds = bd.TextClassificationDataset(enc, labels)

    splits = {
        "train": ds,
        "val": ds,
        "test": ds,
        "label_names": ["sports", "politics", "computers"],
        "tokenizer": tok,
    }

    result = bt.run_finetune(
        model=model,
        splits=splits,
        out_dir=tmp_path,
        lr=5e-4,
        epochs=1,
        batch_size=4,
        run_name="smoke-test",
        report_to=[],
    )

    assert set(result.keys()) >= {"test_metrics", "best_eval_metric"}
    tm = result["test_metrics"]
    assert set(tm.keys()) == {"accuracy", "macro_f1", "per_class_f1", "confusion_matrix"}
    assert tm["confusion_matrix"].shape == (3, 3)
