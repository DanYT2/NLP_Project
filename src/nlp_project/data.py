"""Data layer for Q1: loading and preprocessing 20 Newsgroups.

The dataset is *never* committed to git — it is fetched at runtime from
sklearn's bundled mirror via :func:`load_20ng`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def load_20ng(
    remove: bool = True,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    """Fetch the 20 Newsgroups train/test splits.

    Parameters
    ----------
    remove:
        If True (default), strip headers, footers, and quoted text from
        every document. This is the *honest* setting — leaving headers in
        leaks the label (see spec §3, "label leakage").

    Returns
    -------
    (train_docs, train_labels, test_docs, test_labels, label_names)
        Documents are raw strings; labels are integer arrays in [0, 20);
        label_names is the 20-string list of newsgroup names.
    """
    remove_tuple = ("headers", "footers", "quotes") if remove else ()
    train = fetch_20newsgroups(subset="train", remove=remove_tuple)
    test = fetch_20newsgroups(subset="test", remove=remove_tuple)
    return (
        list(train.data),
        np.asarray(train.target),
        list(test.data),
        np.asarray(test.target),
        list(train.target_names),
    )


@lru_cache(maxsize=1)
def _default_stopwords() -> frozenset[str]:
    """Load NLTK English stopwords once. Requires `scripts/setup_nltk.py`
    to have been run, otherwise raises a clear LookupError.
    """
    from nltk.corpus import stopwords  # local import keeps the package
                                       # importable without nltk data.
    return frozenset(stopwords.words("english"))


def preprocess(
    docs: Sequence[str],
    drop_stopwords: bool = True,
    stopwords: Sequence[str] | set[str] | None = None,
    min_token_len: int = 3,
) -> list[list[str]]:
    """Tokenize, lowercase, drop short tokens, optionally drop stopwords.

    Parameters
    ----------
    docs:
        Raw document strings.
    drop_stopwords:
        If True (default), filter tokens that appear in ``stopwords``.
        Disable this when feeding tokens to word2vec — embeddings learn
        better when frequent function words remain in the context window.
    stopwords:
        Override the default NLTK English stopword list. Pass a set in
        tests so they don't depend on NLTK's data being downloaded.
    min_token_len:
        Drop tokens shorter than this (default 3). gensim's
        ``simple_preprocess`` already lowercases and strips punctuation.
    """
    if drop_stopwords:
        sw = set(stopwords) if stopwords is not None else set(_default_stopwords())
    else:
        sw = set()

    out: list[list[str]] = []
    for doc in docs:
        toks = simple_preprocess(doc, min_len=min_token_len)
        if sw:
            toks = [t for t in toks if t not in sw]
        out.append(toks)
    return out


def train_val_split(
    docs: Sequence[str] | list[list[str]],
    labels: np.ndarray,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list, np.ndarray, list, np.ndarray]:
    """Stratified train/validation split.

    Works on either a list of raw strings or a list of token lists.
    """
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        list(docs),
        labels,
        test_size=val_frac,
        stratify=labels,
        random_state=seed,
    )
    return train_docs, train_labels, val_docs, val_labels
