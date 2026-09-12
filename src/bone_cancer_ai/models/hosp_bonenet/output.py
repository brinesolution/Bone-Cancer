"""Typed HOSP-BoneNet forward output."""

from typing import TypedDict

import torch


class HOSPOutput(TypedDict, total=False):
    """Tensor mapping whose optional keys follow the model configuration."""

    binary_logits: torch.Tensor
    subtype_logits: torch.Tensor
    risk_logits: torch.Tensor
    concept_logits: torch.Tensor
    concept_maps: torch.Tensor
    lesion_logits: torch.Tensor
    global_features: torch.Tensor
    local_features: torch.Tensor
    fused_features: torch.Tensor


__all__ = ["HOSPOutput"]
