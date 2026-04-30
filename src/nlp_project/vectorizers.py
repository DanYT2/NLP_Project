"""TF-IDF vectorization for Q1c.

Thin wrapper around scikit-learn's TfidfVectorizer that fixes the
project's defaults in one place. Inputs are document *strings*, not
token lists — TfidfVectorizer does its own tokenization. Pass
already-preprocessed strings (e.g. ``" ".join(tokens)``) to keep the
preprocessing consistent with the word2vec input.
"""

from __future__ import annotations

from typing import Sequence

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(
    docs: Sequence[str],
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit a TfidfVectorizer on ``docs`` and return both it and the matrix."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    X = vec.fit_transform(docs)
    return vec, X


def transform_tfidf(vec: TfidfVectorizer, docs: Sequence[str]) -> sp.csr_matrix:
    """Apply a fitted vectorizer to new documents."""
    return vec.transform(docs)
