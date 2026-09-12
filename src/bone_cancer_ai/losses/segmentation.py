"""Masked Dice and binary cross-entropy segmentation loss."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as F

from .classification import _batch_vector, differentiable_zero


class SegmentationLoss(NamedTuple):
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor


def _segmentation_tensors(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("lesion logits must be a floating-point tensor")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("lesion logits must have shape [batch, 1, height, width]")
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if targets.shape != logits.shape:
        raise ValueError("segmentation targets must match the lesion logits shape")
    return logits, targets.to(device=logits.device, dtype=logits.dtype)


def dice_bce_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> SegmentationLoss:
    """Return BCE + soft Dice over explicitly valid masks."""
    if smooth <= 0:
        raise ValueError("smooth must be positive")
    logits, targets = _segmentation_tensors(logits, targets)
    size = logits.shape[0]
    valid_mask = _batch_vector(valid_mask, name="segmentation valid mask", size=size)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        zero = differentiable_zero(logits)
        return SegmentationLoss(zero, zero, zero)
    selected_logits = logits[valid].float()
    selected_targets = targets[valid].float()
    if not bool(((selected_targets == 0) | (selected_targets == 1)).all()):
        raise ValueError("valid segmentation targets must contain only 0 and 1")
    bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets)
    probabilities = selected_logits.sigmoid().flatten(start_dim=1)
    flattened_targets = selected_targets.flatten(start_dim=1)
    intersection = (probabilities * flattened_targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + flattened_targets.sum(dim=1)
    dice = (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()
    return SegmentationLoss(bce + dice, bce, dice)


segmentation_loss = dice_bce_segmentation_loss
