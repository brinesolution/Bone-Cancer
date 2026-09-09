"""Conservative, non-destructive screening of decoded image pixels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from bone_cancer_ai.data.schema import QualityStatus


def analyze_pixels(image: Image.Image, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Measure a small grayscale preview and flag uniform scans for manual review."""
    size = int(thresholds.get("thumbnail_size", 256))
    preview = image.copy()
    preview.thumbnail((size, size), Image.Resampling.BILINEAR)
    rgb_grayscale = False
    if image.mode in {"RGB", "RGBA"}:
        rgb = np.asarray(preview.convert("RGB"), dtype=np.int16)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        rgb_grayscale = bool(spread.max(initial=0) <= thresholds.get("rgb_grayscale_tolerance", 1))
    values = np.asarray(preview.convert("L"), dtype=np.uint8)
    mean = float(values.mean())
    std = float(values.std())
    counts = np.bincount(values.ravel(), minlength=256)
    probabilities = counts[counts > 0] / values.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    dark = float(np.mean(values <= thresholds.get("dark_pixel_threshold", 5)))
    bright = float(np.mean(values >= thresholds.get("bright_pixel_threshold", 250)))
    if std <= thresholds.get("blank_std_threshold", 0.5) or entropy <= thresholds.get(
        "blank_entropy_threshold", 0.1
    ):
        status = QualityStatus.BLANK
    elif std <= thresholds.get("near_blank_std_threshold", 3.0) or entropy <= thresholds.get(
        "near_blank_entropy_threshold", 1.0
    ):
        status = QualityStatus.NEAR_BLANK
    else:
        status = QualityStatus.READABLE
    return {
        "is_rgb_encoded_grayscale": rgb_grayscale,
        "intensity_mean": mean,
        "intensity_std": std,
        "entropy": entropy,
        "dark_fraction": dark,
        "bright_fraction": bright,
        "quality_status": str(status),
    }


def build_quality_report(master: pd.DataFrame) -> pd.DataFrame:
    """Select per-image quality evidence without making exclusion decisions."""
    columns = [
        "record_id",
        "image_id",
        "official_split",
        "width",
        "height",
        "channels",
        "pil_mode",
        "is_rgb_encoded_grayscale",
        "intensity_mean",
        "intensity_std",
        "entropy",
        "dark_fraction",
        "bright_fraction",
        "quality_status",
        "decode_error",
    ]
    return master.reindex(columns=columns).copy()
