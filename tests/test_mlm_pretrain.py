"""Tests for src/nlp_project/mlm_pretrain.py.

The slow smoke test deliberately builds a *tiny* BertForMaskedLM from a fresh
BertConfig so the test never has to download multi-hundred-MB model weights
from the Hub. We reuse the real bert-base-uncased tokenizer (small download,
cached after Q2 tests).
"""

from __future__ import annotations

import math

import pytest

from nlp_project import mlm_pretrain as mp


# ---------- fast unit tests ----------


def test_make_mlm_training_args_propagates_hparams(tmp_path) -> None:
    args = mp.make_mlm_training_args(
        out_dir=tmp_path,
        lr=5e-5,
        epochs=2,
        batch_size=8,
        run_name="unit-mlm",
    )
    assert args.learning_rate == 5e-5
    assert args.num_train_epochs == 2
    assert args.per_device_train_batch_size == 8
    assert args.per_device_eval_batch_size == 8
    assert args.run_name == "unit-mlm"


def test_make_mlm_training_args_optimises_eval_loss(tmp_path) -> None:
    """Lower MLM loss is better — the Trainer must pick the best checkpoint
    by ``eval_loss`` (not by accuracy or F1)."""
    args = mp.make_mlm_training_args(
        out_dir=tmp_path, lr=5e-5, epochs=1, batch_size=4, run_name="x",
    )
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False


def test_make_mlm_training_args_has_mps_safe_flags(tmp_path) -> None:
    """fp16/bf16 must be off on Apple Silicon; pin_memory must be off."""
    args = mp.make_mlm_training_args(
        out_dir=tmp_path, lr=5e-5, epochs=1, batch_size=4, run_name="x",
    )
    assert args.fp16 is False
    assert args.bf16 is False
    assert args.dataloader_pin_memory is False


def test_build_mlm_dataset_returns_expected_keys_and_length() -> None:
    """Synthetic tokenizer-free smoke: pass pre-built encoding dicts via the
    private helper path so this test stays fast (no transformers import)."""
    # We exercise the public function with a fake tokenizer that mimics the
    # part of the HF tokenizer API we actually call. Keeps the unit test
    # offline + sub-second.
    class _FakeTokenizer:
        def __call__(
            self, docs, *, truncation, max_length, padding,
        ):  # noqa: ANN001
            assert truncation is True
            assert padding is False
            return {
                "input_ids": [list(range(min(len(d.split()), max_length))) for d in docs],
                "attention_mask": [
                    [1] * min(len(d.split()), max_length) for d in docs
                ],
            }

    tok = _FakeTokenizer()
    docs = ["hello world", "the quick brown fox jumps"]
    ds = mp.build_mlm_dataset(tok, docs, max_length=4)
    assert len(ds) == 2
    item = ds[0]
    assert set(item.keys()) >= {"input_ids", "attention_mask"}
    # The dataset must NOT carry labels — MLM labels are produced on the fly
    # by DataCollatorForLanguageModeling.
    assert "labels" not in item


# ---------- slow integration smoke test ----------


@pytest.mark.slow
def test_run_mlm_pretrain_smoke(tmp_path) -> None:
    """End-to-end: tiny BertForMaskedLM trained 1 epoch on a toy corpus.

    Verifies the Trainer wires up cleanly, the checkpoint is saved, and the
    returned dict shape includes the perplexity = exp(eval_loss) identity.
    We don't make claims about loss values — the model is randomly init'd
    and the data is tiny.
    """
    from transformers import AutoTokenizer, BertConfig, BertForMaskedLM

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    cfg = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = BertForMaskedLM(cfg)

    docs = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subfield of artificial intelligence",
        "newsgroups are an early form of online discussion",
        "masked language modelling is the bert pretraining objective",
        "natural language processing deals with computational linguistics",
        "the cat sat on the mat very quietly",
        "training neural networks requires careful hyperparameter tuning",
        "the model predicts masked tokens from surrounding context",
        "tokenization splits raw text into wordpiece subword units",
        "deep learning has revolutionised computer vision and nlp",
        "the dataset contains twenty different newsgroup categories",
        "evaluation metrics include accuracy macro f1 and confusion matrix",
        "domain adaptive pretraining can improve downstream task performance",
        "fine tuning adjusts the encoder weights for a specific objective",
        "the apple m series uses metal performance shaders for acceleration",
        "validation loss is monitored for early stopping during training",
    ]

    result = mp.run_mlm_pretrain(
        tokenizer=tok,
        train_docs=docs,
        out_dir=tmp_path,
        model=model,
        lr=5e-4,
        epochs=1,
        batch_size=4,
        max_length=16,
        mlm_probability=0.15,
        run_name="smoke-mlm",
        report_to=[],
    )

    assert set(result.keys()) >= {"out_dir", "log_history", "best_eval_loss", "perplexity"}
    assert (tmp_path / "config.json").exists()
    # Weights file may be either pytorch_model.bin or model.safetensors
    weight_files = list(tmp_path.glob("*.bin")) + list(tmp_path.glob("*.safetensors"))
    assert weight_files, "MLM checkpoint must contain a weights file"
    # Perplexity must be the exponentiation of best_eval_loss.
    assert math.isclose(
        result["perplexity"], math.exp(result["best_eval_loss"]), rel_tol=1e-6,
    )
