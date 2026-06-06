"""Evaluation metrics and confusion-matrix plotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader


def metrics_from_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    label_names: list[str],
) -> dict:
    """Compute the standard classification metrics from already-computed predictions.

    Returns a dict with keys ``accuracy``, ``macro_f1``, ``per_class_f1``
    (length ``len(label_names)``) and ``confusion_matrix``. Both per-class F1
    and the confusion matrix are computed against the full label range so
    classes that never appear in ``y_pred`` or ``y_true`` still occupy a slot.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(label_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_names: list[str],
    device: str,
) -> dict:
    """Run ``model`` over ``loader`` and return metrics.

    Returns a dict with keys ``accuracy``, ``macro_f1``, ``per_class_f1``
    (length ``len(label_names)``) and ``confusion_matrix``.
    """
    model.to(device).eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(yb.tolist())

    return metrics_from_predictions(targets, preds, label_names)


def plot_confusion(
    cm: np.ndarray,
    label_names: list[str],
    save_path: Path | str,
    title: str = "Confusion matrix",
) -> None:
    """Render a confusion-matrix heatmap and save it as PNG."""
    fig, ax = plt.subplots(figsize=(0.5 * len(label_names) + 2, 0.5 * len(label_names) + 2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
