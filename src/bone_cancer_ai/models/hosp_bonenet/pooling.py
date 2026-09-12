"""Global and lesion-weighted spatial pooling."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GlobalPool(nn.Module):
    """Pool a feature map with adaptive averaging or learnable GeM."""

    def __init__(self, method: str = "adaptive_avg", gem_p: float = 3.0) -> None:
        super().__init__()
        aliases = {"avg": "adaptive_avg", "average": "adaptive_avg", "gem": "gem"}
        self.method = aliases.get(method, method)
        if self.method not in {"adaptive_avg", "gem"}:
            raise ValueError("pooling must be 'adaptive_avg' or 'gem'")
        if gem_p <= 0:
            raise ValueError("gem_p must be positive")
        self.gem_p = nn.Parameter(torch.tensor(float(gem_p))) if self.method == "gem" else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one vector per feature map."""

        if self.method == "adaptive_avg":
            return F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        assert self.gem_p is not None
        exponent = self.gem_p.clamp(min=0.1)
        pooled = F.adaptive_avg_pool2d(features.clamp_min(1e-6).pow(exponent), 1)
        return pooled.pow(exponent.reciprocal()).flatten(1)


class MaskWeightedPool(nn.Module):
    """Average features using a soft lesion probability map as spatial weights."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    def forward(self, features: torch.Tensor, lesion_logits: torch.Tensor) -> torch.Tensor:
        """Return stable lesion-weighted feature vectors."""

        if features.ndim != 4 or lesion_logits.ndim != 4:
            raise ValueError("features and lesion_logits must be four-dimensional")
        weights = torch.sigmoid(lesion_logits)
        if weights.shape[-2:] != features.shape[-2:]:
            weights = F.interpolate(
                weights,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        output_dtype = features.dtype
        stable_features = features.float()
        stable_weights = weights.float()
        numerator = (stable_features * stable_weights).sum(dim=(-2, -1))
        denominator = stable_weights.sum(dim=(-2, -1)).clamp_min(self.eps)
        return (numerator / denominator).to(dtype=output_dtype)


MaskWeightedPooling = MaskWeightedPool

__all__ = ["GlobalPool", "MaskWeightedPool", "MaskWeightedPooling"]
