"""Deterministic stratified bootstrap intervals for binary metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch

from .classification import binary_classification_metrics

DEFAULT_BOOTSTRAP_METRICS = ("AUROC", "AUPRC", "balanced_accuracy", "F1")


def bootstrap_confidence_intervals(
    logits: object,
    targets: object,
    *,
    metrics: Iterable[str] = DEFAULT_BOOTSTRAP_METRICS,
    resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Estimate percentile intervals using class-stratified resampling."""
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not isinstance(confidence, float | int) or isinstance(confidence, bool):
        raise ValueError("confidence must be between 0 and 1")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    logit_array = torch.as_tensor(logits, dtype=torch.float64).detach().cpu().numpy()
    target_array = torch.as_tensor(targets, dtype=torch.long).detach().cpu().numpy()
    base = binary_classification_metrics(logit_array, target_array)
    selected = tuple(str(metric) for metric in metrics)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("metrics must be a nonempty unique collection")
    unknown = set(selected) - set(base)
    if unknown:
        raise ValueError(f"unknown bootstrap metrics: {sorted(unknown)}")

    class_indices = [np.flatnonzero(target_array == label) for label in (0, 1)]
    if any(indices.size == 0 for indices in class_indices):
        raise ValueError("bootstrap intervals require both binary classes")
    rng = np.random.default_rng(int(seed))
    sampled: dict[str, list[float]] = {metric: [] for metric in selected}
    for _ in range(resamples):
        indices = np.concatenate(
            [rng.choice(values, size=values.size, replace=True) for values in class_indices]
        )
        values = binary_classification_metrics(logit_array[indices], target_array[indices])
        for metric in selected:
            value = values[metric]
            if value is not None and math.isfinite(float(value)):
                sampled[metric].append(float(value))

    tail = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float | int]] = {}
    for metric in selected:
        estimate = base[metric]
        values = np.asarray(sampled[metric], dtype=np.float64)
        if estimate is None or values.size == 0:
            raise ValueError(f"metric {metric} is undefined for bootstrap inference")
        result[metric] = {
            "estimate": float(estimate),
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
            "confidence": confidence,
            "valid_resamples": int(values.size),
            "total_resamples": resamples,
        }
    return result


__all__ = ["DEFAULT_BOOTSTRAP_METRICS", "bootstrap_confidence_intervals"]
