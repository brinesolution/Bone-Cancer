"""Spatially grounded concept prediction head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConceptHead(nn.Module):
    """Produce per-concept activation maps and spatially pooled logits."""

    def __init__(self, in_channels: int, num_concepts: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        self.activation = nn.Conv2d(in_channels, num_concepts, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled concept logits and input-aligned activation maps."""

        low_resolution_maps = self.activation(features)
        logits = F.adaptive_avg_pool2d(low_resolution_maps, output_size=1).flatten(1)
        maps = F.interpolate(
            low_resolution_maps,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits, maps


__all__ = ["ConceptHead"]
