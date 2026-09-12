"""Masked classification losses for partially supervised batches."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _batch_vector(value: torch.Tensor, *, name: str, size: int) -> torch.Tensor:
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1 or value.shape[0] != size:
        raise ValueError(f"{name} must have shape [{size}]")
    return value


def differentiable_zero(*values: torch.Tensor) -> torch.Tensor:
    """Return a finite scalar zero connected to every supplied tensor."""
    if not values:
        return torch.tensor(0.0)
    reference = values[0]
    zero = reference.new_zeros((), dtype=torch.float32)
    for value in values:
        if value.device != reference.device:
            raise ValueError("prediction tensors must be on the same device")
        if not value.is_floating_point():
            raise TypeError("prediction tensors must use a floating-point dtype")
        finite_value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        zero = zero + finite_value.float().sum() * 0.0
    return zero


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy over explicitly valid samples only."""
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("binary logits must be a floating-point tensor")
    size = logits.shape[0] if logits.ndim > 0 else 0
    logits = _batch_vector(logits, name="binary logits", size=size)
    targets = _batch_vector(targets, name="binary targets", size=size)
    valid_mask = _batch_vector(valid_mask, name="binary valid mask", size=size)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return differentiable_zero(logits)
    selected_targets = targets.to(device=logits.device)[valid]
    if not bool(((selected_targets == 0) | (selected_targets == 1)).all()):
        raise ValueError("valid binary targets must contain only 0 and 1")
    return F.binary_cross_entropy_with_logits(
        logits[valid], selected_targets.to(dtype=logits.dtype)
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Multiclass cross-entropy over explicitly valid samples only."""
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("classification logits must be a floating-point tensor")
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("classification logits must have shape [batch, classes>=2]")
    size = logits.shape[0]
    targets = _batch_vector(targets, name="classification targets", size=size)
    valid_mask = _batch_vector(valid_mask, name="classification valid mask", size=size)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return differentiable_zero(logits)
    selected_targets = targets.to(device=logits.device)[valid]
    if selected_targets.is_floating_point():
        if not bool((selected_targets == selected_targets.round()).all()):
            raise ValueError("valid classification targets must be integer class indices")
    selected_targets = selected_targets.to(dtype=torch.long)
    if not bool(((selected_targets >= 0) & (selected_targets < logits.shape[1])).all()):
        raise ValueError("valid classification targets are outside the logits class range")
    return F.cross_entropy(logits[valid], selected_targets)
