"""Epoch-level linear-warmup and cosine-decay learning-rate schedule."""

from __future__ import annotations

import math

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm linearly to the configured LR, then decay it to zero."""

    if type(total_epochs) is not int or total_epochs < 1:
        raise ValueError("total_epochs must be a positive integer")
    if type(warmup_epochs) is not int or not 0 <= warmup_epochs <= total_epochs:
        raise ValueError("warmup_epochs must be an integer from zero to total_epochs")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return float((epoch_index + 1) / warmup_epochs)
        if warmup_epochs == 0:
            if total_epochs == 1:
                return 1.0
            progress = min(max(epoch_index / (total_epochs - 1), 0.0), 1.0)
            return float(0.5 * (1.0 + math.cos(math.pi * progress)))
        decay_epochs = total_epochs - warmup_epochs
        if decay_epochs == 0:
            return 1.0
        decay_step = epoch_index - warmup_epochs + 1
        progress = min(max(decay_step / decay_epochs, 0.0), 1.0)
        return float(0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
