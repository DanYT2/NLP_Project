"""Data layer for Q1: loading and preprocessing 20 Newsgroups.

The dataset is *never* committed to git — it is fetched at runtime from
sklearn's bundled mirror via :func:`load_20ng`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.datasets import fetch_20newsgroups


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
