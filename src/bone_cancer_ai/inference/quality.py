"""Input image quality checks used before research inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageQualityReport:
    """Small, label-free quality report for one uploaded image."""

    height: int
    width: int
    mean: float
    standard_deviation: float
    dynamic_range: int
    entropy: float
    status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "dynamic_range": self.dynamic_range,
            "entropy": self.entropy,
            "status": self.status,
            "warnings": list(self.warnings),
        }


def assess_image_quality(
    image: np.ndarray,
    *,
    minimum_dimension: int = 32,
    low_contrast_std: float = 3.0,
    low_entropy: float = 1.0,
) -> ImageQualityReport:
    """Assess dimensions and pixel information without using model labels."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy array")
    if image.ndim == 2:
        pixels = image
    elif image.ndim == 3 and image.shape[2] == 3:
        pixels = image.mean(axis=2)
    else:
        raise ValueError("image must have shape [H,W] or [H,W,3]")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    histogram, _ = np.histogram(pixels, bins=256, range=(0, 255), density=False)
    probabilities = histogram.astype(np.float64) / max(1, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = max(0.0, float(-(probabilities * np.log2(probabilities)).sum()))
    standard_deviation = float(pixels.std())
    dynamic_range = int(pixels.max() - pixels.min())
    warnings: list[str] = []
    if min(height, width) < int(minimum_dimension):
        warnings.append("small_image")
    if standard_deviation < float(low_contrast_std) or dynamic_range < 8:
        warnings.append("low_contrast")
    if entropy < float(low_entropy):
        warnings.append("low_entropy")
    status = "invalid" if "small_image" in warnings else ("review" if warnings else "ok")
    return ImageQualityReport(
        height=height,
        width=width,
        mean=float(pixels.mean()),
        standard_deviation=standard_deviation,
        dynamic_range=dynamic_range,
        entropy=entropy,
        status=status,
        warnings=tuple(warnings),
    )


__all__ = ["ImageQualityReport", "assess_image_quality"]
