"""Multi-stage image encoder used by HOSP-BoneNet."""

from __future__ import annotations

import timm
import torch
from torch import nn


class TimmFeatureEncoder(nn.Module):
    """Expose spatial feature stages and their metadata from a timm backbone."""

    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True) -> None:
        super().__init__()
        if not backbone:
            raise ValueError("backbone must be a nonempty timm model name")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )
        self.feature_channels = tuple(int(value) for value in self.backbone.feature_info.channels())
        self.feature_reductions = tuple(
            int(value) for value in self.backbone.feature_info.reduction()
        )
        if not self.feature_channels or len(self.feature_channels) != len(
            self.feature_reductions
        ):
            raise ValueError("backbone must expose matching multi-stage feature metadata")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return encoder feature maps from shallowest to deepest stage."""

        return tuple(self.backbone(images))


FeatureEncoder = TimmFeatureEncoder

__all__ = ["FeatureEncoder", "TimmFeatureEncoder"]
