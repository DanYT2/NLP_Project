"""Tests for src/nlp_project/eval.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import eval as ev


class _ConstantModel(nn.Module):
    """A model that always predicts class ``cls``."""

    def __init__(self, num_classes: int, cls: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls = cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full((x.size(0), self.num_classes), -1e9)
        out[:, self.cls] = 1e9
        return out


def _loader(y: list[int], num_features: int = 4) -> DataLoader:
    X = torch.zeros(len(y), num_features)
    return DataLoader(TensorDataset(X, torch.tensor(y, dtype=torch.long)), batch_size=4)


def test_evaluate_perfect_predictions() -> None:
    model = _ConstantModel(num_classes=3, cls=2)
    loader = _loader([2, 2, 2, 2])
    metrics = ev.evaluate(model, loader, label_names=["a", "b", "c"], device="cpu")
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] > 0.0
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_evaluate_all_wrong() -> None:
    model = _ConstantModel(num_classes=3, cls=0)
    loader = _loader([1, 2, 1, 2])
    metrics = ev.evaluate(model, loader, label_names=["a", "b", "c"], device="cpu")
    assert metrics["accuracy"] == 0.0


def test_plot_confusion_writes_a_png(tmp_path: Path) -> None:
    cm = np.array([[5, 1], [0, 4]])
    out = tmp_path / "cm.png"
    ev.plot_confusion(cm, label_names=["x", "y"], save_path=out)
    assert out.exists() and out.stat().st_size > 0
