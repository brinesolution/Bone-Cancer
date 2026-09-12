"""Learned global-local feature fusion."""

from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Combine global and local vectors with a learned elementwise gate."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.gate = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        """Return ``gate * local + (1 - gate) * global``."""

        if global_features.shape != local_features.shape:
            raise ValueError("global_features and local_features must have identical shapes")
        gate = torch.sigmoid(self.gate(torch.cat((global_features, local_features), dim=1)))
        return gate * local_features + (1.0 - gate) * global_features


GlobalLocalFusion = GatedFusion

__all__ = ["GatedFusion", "GlobalLocalFusion"]
