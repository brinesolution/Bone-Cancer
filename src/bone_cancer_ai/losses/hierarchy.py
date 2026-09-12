"""Explicit consistency loss between binary and configured subtype outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .classification import _batch_vector, differentiable_zero


def hierarchy_consistency_loss(
    binary_logits: torch.Tensor,
    subtype_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_subtype_indices: Sequence[int],
) -> torch.Tensor:
    """Match tumour probability to probability mass of explicitly mapped subtypes."""

    if subtype_logits.ndim != 2 or subtype_logits.shape[1] < 2:
        raise ValueError("subtype logits must have shape [batch, classes>=2]")
    batch_size = subtype_logits.shape[0]
    binary_logits = _batch_vector(binary_logits, name="binary logits", size=batch_size)
    valid_mask = _batch_vector(valid_mask, name="hierarchy valid mask", size=batch_size)
    indices = tuple(dict.fromkeys(int(index) for index in positive_subtype_indices))
    if not indices or min(indices) < 0 or max(indices) >= subtype_logits.shape[1]:
        raise ValueError("positive subtype indices must identify configured subtype classes")
    valid = valid_mask.to(device=binary_logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return differentiable_zero(binary_logits, subtype_logits)
    subtype_probability = subtype_logits.to(binary_logits.device).softmax(dim=1)[
        :, list(indices)
    ].sum(dim=1)
    return F.mse_loss(binary_logits[valid].sigmoid(), subtype_probability[valid])


__all__ = ["hierarchy_consistency_loss"]
