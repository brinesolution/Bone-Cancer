"""Calibration and uncertainty helpers for binary predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_calibration_curve(
    probabilities: object, targets: object, *, bins: int = 10
) -> pd.DataFrame:
    """Return nonempty equal-width calibration bins with counts."""
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(targets, dtype=np.int64)
    if probability.ndim != 1 or target.ndim != 1 or probability.shape != target.shape:
        raise ValueError("probabilities and targets must be matching one-dimensional arrays")
    if probability.size == 0 or not np.isfinite(probability).all():
        raise ValueError("probabilities must be finite and nonempty")
    if not np.isin(target, [0, 1]).all() or not (
        (0.0 <= probability) & (probability <= 1.0)
    ).all():
        raise ValueError("targets must be binary and probabilities must be in [0, 1]")
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer of at least 2")
    assignments = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            rows.append(
                {
                    "bin": index,
                    "count": int(selected.sum()),
                    "mean_probability": float(probability[selected].mean()),
                    "observed_frequency": float(target[selected].mean()),
                }
            )
    return pd.DataFrame(rows)


def binary_entropy(probabilities: object) -> np.ndarray:
    """Return binary predictive entropy normalized to the interval [0, 1]."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    if not ((0.0 <= values) & (values <= 1.0)).all():
        raise ValueError("probabilities must be in [0, 1]")
    clipped = np.clip(values, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return entropy / np.log(2.0)


__all__ = ["binary_calibration_curve", "binary_entropy"]
