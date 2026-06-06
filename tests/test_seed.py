"""Verify set_seed makes Python, NumPy, and torch RNGs reproducible."""

from __future__ import annotations

import random

import numpy as np
import torch

from nlp_project import set_seed


def test_set_seed_makes_python_random_deterministic() -> None:
    set_seed()
    a = [random.random() for _ in range(5)]
    set_seed()
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_deterministic() -> None:
    set_seed()
    a = np.random.rand(5)
    set_seed()
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_makes_torch_deterministic() -> None:
    set_seed()
    a = torch.rand(5)
    set_seed()
    b = torch.rand(5)
    torch.testing.assert_close(a, b)
