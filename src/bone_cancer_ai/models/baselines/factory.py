"""Factory helpers for the common Phase 5 baseline API."""

from torch import nn

from bone_cancer_ai.models.baselines.timm_classifier import TimmClassifier
from bone_cancer_ai.models.registry import BASELINE_REGISTRY


def create_baseline(
    name: str,
    pretrained: bool = True,
    image_size: int = 384,
) -> TimmClassifier:
    """Create an approved binary baseline by its public alias."""

    try:
        spec = BASELINE_REGISTRY[name]
    except KeyError as error:
        approved = ", ".join(BASELINE_REGISTRY)
        raise ValueError(f"Unknown baseline {name!r}; choose one of: {approved}") from error
    return TimmClassifier(spec=spec, pretrained=pretrained, image_size=image_size)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that will receive gradient updates."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
