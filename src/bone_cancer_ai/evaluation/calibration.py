"""Calibration and uncertainty helpers for binary predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def _tensor_copy(value: object, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


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


@dataclass(frozen=True)
class TemperatureScaler:
    """Deterministic multiclass temperature scaling fit on calibration logits only."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        value = float(self.temperature)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive")

    @staticmethod
    def _validated(logits: object, targets: object) -> tuple[torch.Tensor, torch.Tensor]:
        values = _tensor_copy(logits, dtype=torch.float64)
        labels = _tensor_copy(targets, dtype=torch.long)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
            raise ValueError("logits must be a non-empty [N,C] matrix with C >= 2")
        if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
            raise ValueError("targets must match logits rows")
        if not bool(torch.isfinite(values).all()) or not bool(
            ((labels >= 0) & (labels < values.shape[1])).all()
        ):
            raise ValueError("logits and targets must be finite and in range")
        return values, labels

    @classmethod
    def fit(
        cls,
        logits: object,
        targets: object,
        *,
        minimum: float = 0.05,
        maximum: float = 20.0,
        steps: int = 161,
    ) -> TemperatureScaler:
        """Select the temperature minimizing calibration cross-entropy on held-out rows."""

        values, labels = cls._validated(logits, targets)
        if not 0.0 < float(minimum) < float(maximum):
            raise ValueError("temperature bounds must be positive and ordered")
        if type(steps) is not int or steps < 2:
            raise ValueError("steps must be an integer of at least 2")
        candidates = torch.exp(
            torch.linspace(np.log(float(minimum)), np.log(float(maximum)), steps)
        )
        losses = torch.stack(
            [F.cross_entropy(values / candidate, labels) for candidate in candidates]
        )
        index = int(torch.argmin(losses).item())
        return cls(float(candidates[index].item()))

    def transform_logits(self, logits: object) -> torch.Tensor:
        values = _tensor_copy(logits, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] < 2 or not bool(torch.isfinite(values).all()):
            raise ValueError("logits must be a finite [N,C] matrix with C >= 2")
        return values / float(self.temperature)

    def nll(
        self,
        logits: object,
        targets: object,
        *,
        temperature: float | None = None,
    ) -> float:
        values, labels = self._validated(logits, targets)
        scale = float(self.temperature if temperature is None else temperature)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("temperature must be finite and positive")
        return float(F.cross_entropy(values / scale, labels).item())

    def to_dict(self) -> dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TemperatureScaler:
        if set(value) != {"temperature"}:
            raise ValueError("temperature state must contain only temperature")
        return cls(float(value["temperature"]))


@dataclass(frozen=True)
class BinaryTemperatureScaler:
    """Temperature scaling for one-logit binary predictions."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        value = float(self.temperature)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive")

    @classmethod
    def fit(
        cls,
        logits: object,
        targets: object,
        *,
        minimum: float = 0.05,
        maximum: float = 20.0,
        steps: int = 161,
    ) -> BinaryTemperatureScaler:
        values = _tensor_copy(logits, dtype=torch.float64).reshape(-1)
        labels = _tensor_copy(targets, dtype=torch.float64).reshape(-1)
        if values.numel() == 0 or values.shape != labels.shape:
            raise ValueError("binary logits and targets must be matching and nonempty")
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(labels).all()):
            raise ValueError("binary logits and targets must be finite")
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("binary targets must be 0 or 1")
        if not 0.0 < float(minimum) < float(maximum):
            raise ValueError("temperature bounds must be positive and ordered")
        if type(steps) is not int or steps < 2:
            raise ValueError("steps must be an integer of at least 2")
        candidates = torch.exp(
            torch.linspace(np.log(float(minimum)), np.log(float(maximum)), steps)
        )
        losses = torch.stack(
            [
                F.binary_cross_entropy_with_logits(values / candidate, labels)
                for candidate in candidates
            ]
        )
        return cls(float(candidates[int(torch.argmin(losses).item())].item()))

    def transform_logits(self, logits: object) -> torch.Tensor:
        values = _tensor_copy(logits, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("binary logits must be finite")
        return values / float(self.temperature)

    def nll(
        self,
        logits: object,
        targets: object,
        *,
        temperature: float | None = None,
    ) -> float:
        values = _tensor_copy(logits, dtype=torch.float64).reshape(-1)
        labels = _tensor_copy(targets, dtype=torch.float64).reshape(-1)
        scale = float(self.temperature if temperature is None else temperature)
        return float(F.binary_cross_entropy_with_logits(values / scale, labels).item())

    def to_dict(self) -> dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> BinaryTemperatureScaler:
        if set(value) != {"temperature"}:
            raise ValueError("binary temperature state must contain only temperature")
        return cls(float(value["temperature"]))


__all__ = [
    "BinaryTemperatureScaler",
    "TemperatureScaler",
    "binary_calibration_curve",
    "binary_entropy",
]
