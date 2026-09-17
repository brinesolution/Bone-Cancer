"""Split-conformal prediction sets for calibrated binary probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _probabilities(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("probabilities must be a finite nonempty one-dimensional array")
    if not ((0.0 <= result) & (result <= 1.0)).all():
        raise ValueError("probabilities must be in [0, 1]")
    return result


@dataclass(frozen=True)
class BinaryConformalCalibrator:
    """Least-ambiguous-set conformal calibrator using held-out labels."""

    alpha: float
    qhat: float
    calibration_count: int

    @classmethod
    def fit(
        cls, probabilities: object, targets: object, *, alpha: float = 0.10
    ) -> BinaryConformalCalibrator:
        values = _probabilities(probabilities)
        labels = np.asarray(targets, dtype=np.int64)
        if labels.ndim != 1 or labels.shape != values.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("targets must be matching binary labels")
        if isinstance(alpha, bool) or not isinstance(alpha, float | int):
            raise ValueError("alpha must be between 0 and 1")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        true_probability = np.where(labels == 1, values, 1.0 - values)
        scores = 1.0 - true_probability
        rank = min(scores.size, math.ceil((scores.size + 1) * (1.0 - alpha)))
        qhat = float(np.sort(scores)[rank - 1])
        return cls(alpha=alpha, qhat=qhat, calibration_count=int(scores.size))

    def predict_sets(self, probabilities: object) -> list[list[int]]:
        values = _probabilities(probabilities)
        class_probabilities = np.column_stack((1.0 - values, values))
        included = 1.0 - class_probabilities <= self.qhat + 1e-12
        return [np.flatnonzero(row).astype(int).tolist() for row in included]

    def summarize(self, probabilities: object, targets: object) -> dict[str, float | int]:
        values = _probabilities(probabilities)
        labels = np.asarray(targets, dtype=np.int64)
        if labels.ndim != 1 or labels.shape != values.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("targets must be matching binary labels")
        sets = self.predict_sets(values)
        sizes = np.asarray([len(values_) for values_ in sets], dtype=np.int64)
        covered = np.asarray(
            [int(label) in values_ for label, values_ in zip(labels, sets, strict=True)]
        )
        return {
            "alpha": self.alpha,
            "qhat": self.qhat,
            "calibration_count": self.calibration_count,
            "test_count": int(labels.size),
            "coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "singleton_count": int(np.sum(sizes == 1)),
            "ambiguous_count": int(np.sum(sizes > 1)),
            "empty_count": int(np.sum(sizes == 0)),
        }


__all__ = ["BinaryConformalCalibrator"]
