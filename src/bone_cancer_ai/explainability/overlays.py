"""Image denormalization and spatial heatmap overlays."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps

from bone_cancer_ai.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert one ImageNet-normalized CHW tensor into an RGB float image."""
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device)[:, None, None]
    return (
        (image.detach() * std + mean)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def overlay_heatmap(image: object, heatmap: object, *, alpha: float = 0.40) -> np.ndarray:
    """Blend a normalized heatmap with an RGB image."""
    values = np.asarray(image, dtype=np.float32)
    heat = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if heat.shape != values.shape[:2]:
        raise ValueError("heatmap shape must match the image spatial shape")
    if not np.isfinite(values).all() or not np.isfinite(heat).all():
        raise ValueError("image and heatmap must be finite")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not ((0.0 <= values) & (values <= 1.0)).all():
        raise ValueError("image values must be in [0, 1]")
    heat = np.clip(heat, 0.0, 1.0)
    colors = colormaps["jet"](heat)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * values + alpha * colors, 0.0, 1.0).astype(np.float32)


__all__ = ["denormalize_image", "overlay_heatmap"]
