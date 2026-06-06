"""HSLU NALAPRO Q1 — two-layer MLP on 20 Newsgroups.

Top-level package. All randomness in the project is seeded from SEED.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED: int = 42
"""Single source of truth for randomness across the project."""


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) from one constant.

    Call this at the top of every notebook and every test that depends on
    deterministic behaviour. Word2vec uses a separate `seed=` kwarg on the
    gensim model itself — we set it explicitly there.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is a hard dep, but we don't want this helper to fail
        # if it's ever called from a stripped-down environment.
        pass


__all__ = ["SEED", "set_seed"]
