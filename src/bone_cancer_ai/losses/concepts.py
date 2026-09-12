"""Missing-label-aware concept prediction loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .classification import differentiable_zero


def masked_concept_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply BCE only to explicitly valid concept entries."""

    if not logits.is_floating_point() or logits.ndim != 2:
        raise ValueError("concept logits must be a floating-point [batch, concepts] tensor")
    if targets.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("concept targets and validity mask must match concept logits")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return differentiable_zero(logits)
    selected_targets = targets.to(device=logits.device, dtype=logits.dtype)[valid]
    if not bool(((selected_targets == 0) | (selected_targets == 1)).all()):
        raise ValueError("valid concept targets must contain only 0 and 1")
    return F.binary_cross_entropy_with_logits(logits[valid], selected_targets)


__all__ = ["masked_concept_loss"]
