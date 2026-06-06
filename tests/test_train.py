"""Tests for src/nlp_project/train.py.

Strategy: train on a tiny separable synthetic dataset and assert the
final training accuracy is high. This is the canonical way to test a
training loop end-to-end without depending on real data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import set_seed
from nlp_project.model import MLP
from nlp_project.train import train


def _toy_loaders() -> tuple[DataLoader, DataLoader]:
    """Two well-separated Gaussian clusters in 8 dimensions."""
    set_seed()
    n = 256
    X0 = np.random.randn(n, 8).astype(np.float32) - 2.0
    X1 = np.random.randn(n, 8).astype(np.float32) + 2.0
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n, dtype=np.int64)
    perm = np.random.permutation(2 * n)
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(val_ds, batch_size=32)


def test_train_overfits_separable_synthetic_data() -> None:
    train_loader, val_loader = _toy_loaders()
    model = MLP(in_dim=8, hidden_dim=16, num_classes=2, dropout=0.0)
    history = train(
        model, train_loader, val_loader,
        epochs=20, lr=1e-2, device="cpu", wandb_run=None, patience=20,
    )
    assert history["train_acc"][-1] > 0.95
    assert history["val_acc"][-1] > 0.90


def test_train_returns_history_with_expected_keys() -> None:
    train_loader, val_loader = _toy_loaders()
    model = MLP(in_dim=8, hidden_dim=16, num_classes=2, dropout=0.0)
    history = train(
        model, train_loader, val_loader,
        epochs=2, lr=1e-2, device="cpu", wandb_run=None, patience=10,
    )
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert key in history
        assert len(history[key]) == 2


def test_train_early_stops_on_val_loss() -> None:
    """If val loss never improves, early stopping should trigger before max epochs."""
    # Create a pathological dataset where val loss can't improve.
    # Use separate train/val splits from the noisy data: the model can overfit
    # train labels but val loss stagnates or rises, triggering early stopping.
    set_seed()
    n = 64
    X = np.random.randn(n, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=n, dtype=np.int64)  # pure noise
    train_ds = TensorDataset(torch.from_numpy(X[:48]), torch.from_numpy(y[:48]))
    val_ds = TensorDataset(torch.from_numpy(X[48:]), torch.from_numpy(y[48:]))
    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)
    model = MLP(in_dim=4, hidden_dim=8, num_classes=2, dropout=0.0)
    history = train(
        model, train_loader, val_loader,
        epochs=50, lr=1e-2, device="cpu", wandb_run=None, patience=3,
    )
    # We should not have run all 50 epochs.
    assert len(history["val_loss"]) < 50
