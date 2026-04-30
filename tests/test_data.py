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
