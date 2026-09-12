"""Configured hierarchical classification heads."""

from __future__ import annotations

import torch
from torch import nn


class HierarchicalHeads(nn.Module):
    """Apply the mandatory binary head and configured auxiliary heads."""

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        num_subtypes: int | None = None,
        num_risk_classes: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = nn.Dropout(dropout)
        self.binary = nn.Linear(input_dim, 1)
        self.subtype = nn.Linear(input_dim, num_subtypes) if num_subtypes is not None else None
        self.risk = (
            nn.Linear(input_dim, num_risk_classes)
            if num_risk_classes is not None
            else None
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits only for heads created by configuration."""

        value = self.dropout(features)
        output = {"binary_logits": self.binary(value).squeeze(-1)}
        if self.subtype is not None:
            output["subtype_logits"] = self.subtype(value)
        if self.risk is not None:
            output["risk_logits"] = self.risk(value)
        return output


HierarchyHeads = HierarchicalHeads

__all__ = ["HierarchicalHeads", "HierarchyHeads"]
