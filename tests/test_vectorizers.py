"""Tests for src/nlp_project/vectorizers.py."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from nlp_project import vectorizers


def test_fit_tfidf_returns_sparse_matrix_of_correct_shape() -> None:
    docs = ["the cat sat on the mat", "the dog barked at the cat", "dogs love cats"]
    vec, X = vectorizers.fit_tfidf(docs, max_features=20, min_df=1)
    assert sp.issparse(X)
    assert X.shape[0] == 3
    assert X.shape[1] <= 20


def test_transform_tfidf_uses_train_vocab() -> None:
    train = ["the cat sat", "the dog barked"]
    test = ["a brand new sentence with cat"]
    vec, X_train = vectorizers.fit_tfidf(train, max_features=50, min_df=1)
    X_test = vectorizers.transform_tfidf(vec, test)
    # Test matrix must have the same number of features as train.
    assert X_test.shape[1] == X_train.shape[1]
    # "brand", "new", "sentence", "with" are not in train vocab and
    # should be ignored; only "cat" should yield a non-zero column.
    assert X_test.nnz <= 2  # "cat" plus possibly tf-idf-zero rows
