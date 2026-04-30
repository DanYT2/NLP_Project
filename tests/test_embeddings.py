"""Tests for src/nlp_project/embeddings.py."""

from __future__ import annotations

import numpy as np
import pytest

from nlp_project import embeddings


@pytest.fixture
def tiny_corpus() -> list[list[str]]:
    """A toy corpus rich enough for word2vec to assign vectors.

    Word2vec needs each token to appear at least `min_count` times. We
    use min_count=1 in tests and repeat the same 4-document mini-corpus
    4 times so the vocab stabilises.
    """
    base = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "barked", "at", "the", "cat"],
        ["the", "dog", "and", "the", "cat", "are", "friends"],
        ["dogs", "love", "cats", "and", "cats", "love", "dogs"],
    ]
    return base * 4


def test_train_word2vec_vocab_contains_expected_tokens(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1)
    for tok in ["the", "cat", "dog"]:
        assert tok in model.wv.key_to_index


def test_train_word2vec_vector_size(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1)
    assert model.wv["cat"].shape == (16,)


def test_train_word2vec_is_deterministic_with_workers_1(tiny_corpus) -> None:
    a = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1, seed=42)
    b = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1, seed=42)
    np.testing.assert_array_equal(a.wv["cat"], b.wv["cat"])


def test_mean_pool_shape(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    docs = [["cat", "dog"], ["the", "mat"]]
    X = embeddings.mean_pool(docs, model)
    assert X.shape == (2, 8)
    assert X.dtype == np.float32


def test_mean_pool_matches_hand_computed(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    expected = (model.wv["cat"] + model.wv["dog"]) / 2.0
    X = embeddings.mean_pool([["cat", "dog"]], model)
    np.testing.assert_allclose(X[0], expected, rtol=1e-5)


def test_mean_pool_zero_vector_for_all_oov(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    X = embeddings.mean_pool([["xyzzy", "qux", "frobnicate"]], model)
    assert X.shape == (1, 8)
    np.testing.assert_array_equal(X[0], np.zeros(8, dtype=np.float32))
