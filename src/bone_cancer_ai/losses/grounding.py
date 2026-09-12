"""Spatial grounding loss for verified positive concepts and lesion masks."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .classification import differentiable_zero


def spatial_grounding_loss(
    concept_maps: torch.Tensor,
    lesion_targets: torch.Tensor,
    concept_targets: torch.Tensor,
    concept_valid: torch.Tensor,
    mask_valid: torch.Tensor,
) -> torch.Tensor:
    """Penalize positive concept activation outside a verified lesion region."""

    if not concept_maps.is_floating_point() or concept_maps.ndim != 4:
        raise ValueError("concept maps must be a floating-point [batch, concepts, H, W] tensor")
    batch_size, num_concepts = concept_maps.shape[:2]
    if concept_targets.shape != (batch_size, num_concepts):
        raise ValueError("concept targets must match concept map batch and concept dimensions")
    if concept_valid.shape != concept_targets.shape:
        raise ValueError("concept validity mask must match concept targets")
    if lesion_targets.ndim == 3:
        lesion_targets = lesion_targets.unsqueeze(1)
    if lesion_targets.ndim != 4 or lesion_targets.shape[:2] != (batch_size, 1):
        raise ValueError("lesion targets must have shape [batch, 1, H, W]")
    if mask_valid.shape != (batch_size,):
        raise ValueError("mask validity must have shape [batch]")

    targets = concept_targets.to(device=concept_maps.device)
    valid = (
        concept_valid.to(device=concept_maps.device, dtype=torch.bool)
        & mask_valid.to(device=concept_maps.device, dtype=torch.bool).unsqueeze(1)
        & (targets == 1)
    )
    if not bool(valid.any()):
        return differentiable_zero(concept_maps)

    masks = lesion_targets.to(device=concept_maps.device, dtype=concept_maps.dtype)
    valid_masks = masks[mask_valid.to(device=concept_maps.device, dtype=torch.bool)]
    if not bool(((valid_masks == 0) | (valid_masks == 1)).all()):
        raise ValueError("valid lesion targets must contain only 0 and 1")
    masks = F.interpolate(masks, size=concept_maps.shape[-2:], mode="nearest")
    outside_activation = concept_maps.sigmoid() * (1.0 - masks)
    per_concept = outside_activation.flatten(start_dim=2).mean(dim=2)
    return per_concept[valid].mean()


__all__ = ["spatial_grounding_loss"]
