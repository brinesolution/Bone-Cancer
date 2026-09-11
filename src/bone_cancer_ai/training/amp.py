"""Shared automatic mixed-precision construction for baseline training."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch


def amp_is_enabled(device: torch.device, requested: bool) -> bool:
    """Enable AMP only on the CUDA path used by real baseline runs."""

    return bool(requested and device.type == "cuda")


def autocast_context(
    device: torch.device, requested: bool
) -> AbstractContextManager[object]:
    """Return the modern device-specific PyTorch autocast context."""

    return torch.amp.autocast(
        device_type=device.type,
        enabled=amp_is_enabled(device, requested),
    )


def build_grad_scaler(device: torch.device, requested: bool) -> torch.amp.GradScaler:
    """Create a scaler whose enabled state agrees with the autocast path."""

    return torch.amp.GradScaler(
        device=device.type,
        enabled=amp_is_enabled(device, requested),
        init_scale=64.0,
        growth_interval=100000,
    )
