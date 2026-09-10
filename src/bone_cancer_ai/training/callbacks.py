"""Validation-AUROC early stopping with serializable state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class EarlyStopping:
    """Stop after ``patience`` epochs without a strict metric improvement."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        min_delta = float(min_delta)
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def step(self, value: float, *, epoch: int) -> bool:
        """Update selection state and return whether training should stop."""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError("early-stopping metric must be finite")
        if type(epoch) is not int or epoch < 1:
            raise ValueError("epoch must be a positive integer")
        if self.best_value is None or value > self.best_value + self.min_delta:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        """Return strict JSON-safe callback state for checkpoint resume."""

        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state only when its stopping policy matches this instance."""

        if not isinstance(state, Mapping):
            raise ValueError("early-stopping state must be a mapping")
        try:
            patience = int(state["patience"])
            min_delta = float(state["min_delta"])
            best_value_raw = state["best_value"]
            best_epoch_raw = state["best_epoch"]
            bad_epochs = int(state["bad_epochs"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("early-stopping state is malformed") from error
        if patience != self.patience or min_delta != self.min_delta:
            raise ValueError("early-stopping policy does not match the checkpoint")
        best_value = None if best_value_raw is None else float(best_value_raw)
        best_epoch = None if best_epoch_raw is None else int(best_epoch_raw)
        if best_value is not None and not math.isfinite(best_value):
            raise ValueError("checkpoint best value must be finite")
        if (best_value is None) != (best_epoch is None):
            raise ValueError("checkpoint best value and epoch must both be set or absent")
        if best_epoch is not None and best_epoch < 1:
            raise ValueError("checkpoint best epoch must be positive")
        if bad_epochs < 0:
            raise ValueError("checkpoint bad epoch count cannot be negative")
        self.best_value = best_value
        self.best_epoch = best_epoch
        self.bad_epochs = bad_epochs
