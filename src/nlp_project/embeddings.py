"""word2vec training and pooling utilities.

We pin ``workers=1`` because gensim's multi-threaded training is
non-deterministic even with a fixed seed (threads consume the corpus in
non-deterministic order). Determinism matters here because the spec
requires comparing 1-epoch and 20-epoch checkpoints — only meaningful if
both models start from byte-identical initial weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gensim.models import Word2Vec


def train_word2vec(
    token_lists: Sequence[Sequence[str]],
    epochs: int,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    sg: int = 1,
    seed: int = 42,
    workers: int = 1,
) -> Word2Vec:
    """Train a skip-gram word2vec model on tokenized documents.

    Parameters
    ----------
    epochs:
        Number of training epochs. Pass 1 for the "untrained" snapshot
        and 20 for the "trained" snapshot used by the spec's comparison.
    sg:
        1 = skip-gram, 0 = CBOW. Skip-gram works better for small corpora
        and rare words.
    workers:
        Defaults to 1 for determinism. Bump to 4 for one-off exploration.
    """
    # Word2Vec expects an iterable of iterables of tokens.
    return Word2Vec(
        sentences=list(token_lists),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs,
    )


def mean_pool(
    token_lists: Sequence[Sequence[str]],
    model: Word2Vec,
) -> np.ndarray:
    """Average the token vectors for each document.

    OOV tokens are skipped. Documents with zero in-vocab tokens get a
    zero vector — the alternative (NaN propagation) breaks downstream
    training, and a zero vector at least keeps batch shapes consistent.
    """
    dim = model.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    wv = model.wv
    for i, toks in enumerate(token_lists):
        vecs = [wv[t] for t in toks if t in wv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
        # else: leave as zeros (already initialised).
    return out
