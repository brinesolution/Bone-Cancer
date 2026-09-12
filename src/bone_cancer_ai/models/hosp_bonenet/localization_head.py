"""Lightweight feature-pyramid lesion localization head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalizationHead(nn.Module):
    """Fuse multi-stage features and return one input-aligned lesion logit map."""

    def __init__(self, feature_channels: tuple[int, ...], pyramid_channels: int) -> None:
        super().__init__()
        if not feature_channels or any(channel <= 0 for channel in feature_channels):
            raise ValueError("feature_channels must contain positive channel counts")
        if pyramid_channels <= 0:
            raise ValueError("pyramid_channels must be positive")
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, pyramid_channels, kernel_size=1)
            for channel in feature_channels
        )
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in feature_channels
        )
        self.classifier = nn.Conv2d(pyramid_channels, 1, kernel_size=1)

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Build a top-down pyramid and resize lesion logits to ``output_size``."""

        if len(features) != len(self.lateral):
            raise ValueError(
                f"expected {len(self.lateral)} feature maps, received {len(features)}"
            )
        pyramid: torch.Tensor | None = None
        for feature, lateral, refine in reversed(
            list(zip(features, self.lateral, self.refine, strict=True))
        ):
            level = lateral(feature)
            if pyramid is not None:
                pyramid = F.interpolate(
                    pyramid,
                    size=level.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                level = level + pyramid
            pyramid = refine(level)
        if pyramid is None:  # guarded by constructor validation
            raise RuntimeError("localization pyramid has no feature maps")
        logits = self.classifier(pyramid)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


LesionLocalizationHead = LocalizationHead

__all__ = ["LesionLocalizationHead", "LocalizationHead"]
