"""Approved timm backbones for the Phase 5 baseline comparison."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class BaselineSpec:
    """Immutable construction details for one approved baseline."""

    timm_name: str
    needs_image_size: bool = False


BASELINE_REGISTRY: Mapping[str, BaselineSpec] = MappingProxyType(
    {
        "densenet121": BaselineSpec(timm_name="densenet121"),
        "efficientnet_b3": BaselineSpec(timm_name="efficientnet_b3"),
        "convnext_tiny": BaselineSpec(timm_name="convnext_tiny"),
        "swin_tiny": BaselineSpec(
            timm_name="swin_tiny_patch4_window7_224",
            needs_image_size=True,
        ),
    }
)
