"""The Q1 model: a two-layer MLP.

Architecture is fixed by the spec: Linear -> ReLU -> Linear, with
optional dropout between the two linear layers. Dropout is on by default
because without it the 256-hidden model overfits 20NG hard, especially
on the 20k-feature TF-IDF input. Dropout is *not* a structural change to
the network — both linear layers and the ReLU are still there.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 20,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
