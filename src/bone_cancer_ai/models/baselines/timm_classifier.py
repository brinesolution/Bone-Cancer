"""Common binary classifier wrapper for approved timm backbones."""

import timm
import torch
from torch import nn

from bone_cancer_ai.models.registry import BaselineSpec


class TimmClassifier(nn.Module):
    """Build a timm backbone with a single-logit classification head."""

    def __init__(self, spec: BaselineSpec, pretrained: bool, image_size: int) -> None:
        super().__init__()
        create_kwargs = {"num_classes": 1}
        if spec.needs_image_size:
            create_kwargs["img_size"] = image_size
        self.backbone = timm.create_model(
            spec.timm_name,
            pretrained=pretrained,
            **create_kwargs,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one binary logit per input image."""

        return self.backbone(images).squeeze(-1)
