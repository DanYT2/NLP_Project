"""Tests for src/nlp_project/data.py.

Network-touching tests are marked `slow` so the inner-loop unit suite stays fast.
Run them explicitly with: uv run pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from nlp_project import data


@pytest.mark.slow
def test_load_20ng_returns_expected_shapes() -> None:
    train_docs, train_labels, test_docs, test_labels, label_names = data.load_20ng()
    # The sklearn 20newsgroups train/test split sizes are public and stable.
    assert len(train_docs) == 11_314
    assert len(test_docs) == 7_532
    assert train_labels.shape == (11_314,)
    assert test_labels.shape == (7_532,)
    assert len(label_names) == 20
    # Labels are integer class indices in [0, 20).
    assert train_labels.dtype.kind in "iu"
    assert int(train_labels.min()) == 0
    assert int(train_labels.max()) == 19


@pytest.mark.slow
def test_load_20ng_strips_headers() -> None:
    train_docs, *_ = data.load_20ng(remove=True)
    # When remove=('headers','footers','quotes') is on, no document
    # should start with the canonical "From:" header line.
    assert not any(doc.lstrip().startswith("From:") for doc in train_docs[:200])


@pytest.mark.slow
def test_load_20ng_returns_arrays_not_lists_for_labels() -> None:
    _, train_labels, *_ = data.load_20ng()
    assert isinstance(train_labels, np.ndarray)


def test_preprocess_lowercases_and_tokenizes() -> None:
    docs = ["Hello WORLD foo"]
    out = data.preprocess(docs, drop_stopwords=False, stopwords=set())
    assert out == [["hello", "world", "foo"]]


def test_preprocess_drops_short_tokens() -> None:
    # gensim.simple_preprocess drops tokens with len < min_len (default 2);
    # we additionally enforce len >= 3.
    docs = ["a be cat dog"]
    out = data.preprocess(docs, drop_stopwords=False, stopwords=set())
    assert out == [["cat", "dog"]]


def test_preprocess_drops_stopwords_when_flag_true() -> None:
    docs = ["the cat sat on the mat"]
    out = data.preprocess(docs, drop_stopwords=True, stopwords={"the", "sat", "mat"})
    assert out == [["cat"]]


def test_preprocess_keeps_stopwords_when_flag_false() -> None:
    docs = ["the cat sat"]
    out = data.preprocess(docs, drop_stopwords=False, stopwords={"the"})
    assert out == [["the", "cat", "sat"]]


def test_preprocess_returns_list_of_lists() -> None:
    out = data.preprocess(["one two three", "four five six"], drop_stopwords=False, stopwords=set())
    assert isinstance(out, list)
    assert all(isinstance(doc, list) for doc in out)
    assert len(out) == 2
