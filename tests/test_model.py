"""Tests for src/nlp_project/model.py."""

from __future__ import annotations

import torch

from nlp_project.model import MLP


def test_mlp_forward_shape() -> None:
    model = MLP(in_dim=100, hidden_dim=64, num_classes=20)
    x = torch.randn(8, 100)
    out = model(x)
    assert out.shape == (8, 20)


def test_mlp_has_two_linear_layers_and_a_relu() -> None:
    """The spec mandates Linear -> ReLU -> Linear. Verify the structure."""
    model = MLP(in_dim=10, hidden_dim=8, num_classes=3)
    children = list(model.children())
    # Sequential with: Linear, ReLU, Dropout, Linear (dropout is allowed).
    layer_types = [type(m).__name__ for m in model.net]
    assert layer_types[0] == "Linear"
    assert layer_types[1] == "ReLU"
    assert "Linear" in layer_types[2:]


def test_mlp_dropout_is_active_in_train_mode() -> None:
    """Same input twice in train mode should yield different outputs
    when dropout > 0 — proves dropout is wired up."""
    torch.manual_seed(0)
    model = MLP(in_dim=10, hidden_dim=64, num_classes=3, dropout=0.5)
    model.train()
    x = torch.randn(4, 10)
    a = model(x)
    b = model(x)
    assert not torch.allclose(a, b)


def test_mlp_dropout_is_inactive_in_eval_mode() -> None:
    model = MLP(in_dim=10, hidden_dim=64, num_classes=3, dropout=0.5)
    model.eval()
    x = torch.randn(4, 10)
    a = model(x)
    b = model(x)
    assert torch.allclose(a, b)
