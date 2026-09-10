"""Atomic, alias-validated training checkpoints."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn

from bone_cancer_ai.training.callbacks import EarlyStopping


def _json_copy(value: object, *, name: str) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict JSON-safe data") from error


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    early_stopping: EarlyStopping,
    model_alias: str,
    epoch: int,
    selection_metric: str,
    selection_value: float,
    resolved_config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically replace one checkpoint with complete resume state."""

    destination = Path(path)
    if not model_alias:
        raise ValueError("model_alias must be non-empty")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    selection_value = float(selection_value)
    if not selection_metric or not math.isfinite(selection_value):
        raise ValueError("selection metric and value must be finite and non-empty")
    config_copy = _json_copy(dict(resolved_config), name="resolved_config")
    history_copy = _json_copy([dict(record) for record in history], name="history")
    payload = {
        "model_alias": model_alias,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "early_stopping_state": early_stopping.state_dict(),
        "epoch": epoch,
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "resolved_config": config_copy,
        "history": history_copy,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_model_alias: str,
    map_location: str | torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
) -> dict[str, Any]:
    """Validate identity, restore requested state, and return resume metadata."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    required = {
        "model_alias",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "early_stopping_state",
        "epoch",
        "selection_metric",
        "selection_value",
        "resolved_config",
        "history",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    actual_alias = checkpoint["model_alias"]
    if actual_alias != expected_model_alias:
        raise ValueError(
            f"checkpoint model alias {actual_alias!r} does not match {expected_model_alias!r}"
        )

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if early_stopping is not None:
        early_stopping.load_state_dict(checkpoint["early_stopping_state"])
    return {
        "model_alias": actual_alias,
        "epoch": int(checkpoint["epoch"]),
        "selection_metric": str(checkpoint["selection_metric"]),
        "selection_value": float(checkpoint["selection_value"]),
        "resolved_config": _json_copy(checkpoint["resolved_config"], name="resolved_config"),
        "history": _json_copy(checkpoint["history"], name="history"),
        "early_stopping_state": _json_copy(
            checkpoint["early_stopping_state"], name="early_stopping_state"
        ),
    }
