"""Atomic, alias-validated training checkpoints."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
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


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite numeric data") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _validated_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_model_alias: str,
    expected_resolved_config: Mapping[str, Any] | None,
    expected_selection_metric: str | None,
) -> dict[str, Any]:
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
    if not isinstance(actual_alias, str) or not actual_alias:
        raise ValueError("checkpoint model alias must be a non-empty string")
    if actual_alias != expected_model_alias:
        raise ValueError(
            f"checkpoint model alias {actual_alias!r} does not match {expected_model_alias!r}"
        )
    epoch = _positive_integer(checkpoint["epoch"], name="checkpoint epoch")
    selection_metric = checkpoint["selection_metric"]
    if not isinstance(selection_metric, str) or not selection_metric:
        raise ValueError("checkpoint selection metric must be a non-empty string")
    if expected_selection_metric is not None and selection_metric != expected_selection_metric:
        raise ValueError("checkpoint selection metric does not match")
    selection_value = _finite_number(
        checkpoint["selection_value"], name="checkpoint selection value"
    )

    config_raw = checkpoint["resolved_config"]
    if not isinstance(config_raw, Mapping):
        raise ValueError("checkpoint resolved configuration must be a mapping")
    resolved_config = _json_copy(dict(config_raw), name="resolved_config")
    if expected_resolved_config is not None:
        expected_config = _json_copy(
            dict(expected_resolved_config), name="expected resolved configuration"
        )
        if resolved_config != expected_config:
            raise ValueError("checkpoint resolved configuration does not match")
    configured_epochs = resolved_config.get("epochs")
    if configured_epochs is not None and (
        type(configured_epochs) is not int or configured_epochs < epoch
    ):
        raise ValueError("checkpoint epoch exceeds the resolved configuration")

    history_raw = checkpoint["history"]
    if isinstance(history_raw, str) or not isinstance(history_raw, Sequence):
        raise ValueError("checkpoint history must be a sequence of mappings")
    history: list[dict[str, Any]] = []
    metric_key = f"val_{selection_metric}"
    for expected_epoch, record_raw in enumerate(history_raw, start=1):
        if not isinstance(record_raw, Mapping):
            raise ValueError("checkpoint history records must be mappings")
        record = _json_copy(dict(record_raw), name="history")
        if record.get("epoch") != expected_epoch or type(record.get("epoch")) is not int:
            raise ValueError("checkpoint history epochs must be contiguous and one-based")
        if metric_key not in record:
            raise ValueError(f"checkpoint history is missing {metric_key!r}")
        _finite_number(record[metric_key], name=f"history {metric_key}")
        history.append(record)
    if len(history) != epoch:
        raise ValueError("checkpoint history length must match its epoch")
    if _finite_number(history[-1][metric_key], name=f"history {metric_key}") != selection_value:
        raise ValueError("checkpoint selection value does not match final history")

    stopping_raw = checkpoint["early_stopping_state"]
    if not isinstance(stopping_raw, Mapping):
        raise ValueError("checkpoint early-stopping state must be a mapping")
    stopping_state = _json_copy(dict(stopping_raw), name="early_stopping_state")
    patience = _positive_integer(
        stopping_state.get("patience"), name="checkpoint early-stopping patience"
    )
    min_delta = _finite_number(
        stopping_state.get("min_delta"), name="checkpoint early-stopping min_delta"
    )
    if min_delta < 0:
        raise ValueError("checkpoint early-stopping min_delta must be non-negative")
    replayed = EarlyStopping(patience=patience, min_delta=min_delta)
    for record in history:
        stopped = replayed.step(float(record[metric_key]), epoch=int(record["epoch"]))
        if stopped and record["epoch"] != epoch:
            raise ValueError("checkpoint history continues after early stopping")
    if replayed.state_dict() != stopping_state:
        raise ValueError("checkpoint early-stopping state does not match history")

    return {
        "model_alias": actual_alias,
        "epoch": epoch,
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "resolved_config": resolved_config,
        "history": history,
        "early_stopping_state": stopping_state,
    }


def _require_state_mapping(checkpoint: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    state = checkpoint[name]
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return state


def _validate_model_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    current = model.state_dict()
    if current.keys() != state.keys():
        raise ValueError("checkpoint model state keys do not match")
    for name, current_value in current.items():
        loaded_value = state[name]
        if not isinstance(loaded_value, torch.Tensor):
            raise ValueError(f"checkpoint model state {name!r} must be a tensor")
        if current_value.shape != loaded_value.shape or current_value.dtype != loaded_value.dtype:
            raise ValueError(f"checkpoint model state {name!r} is incompatible")


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer, state: Mapping[str, Any]
) -> None:
    loaded_groups = state.get("param_groups")
    loaded_state = state.get("state")
    current_groups = optimizer.state_dict()["param_groups"]
    if not isinstance(loaded_groups, list) or not isinstance(loaded_state, Mapping):
        raise ValueError("checkpoint optimizer state is malformed")
    if len(loaded_groups) != len(current_groups):
        raise ValueError("checkpoint optimizer parameter groups do not match")
    parameter_ids: set[object] = set()
    for loaded_group, current_group, live_group in zip(
        loaded_groups, current_groups, optimizer.param_groups, strict=True
    ):
        if not isinstance(loaded_group, Mapping):
            raise ValueError("checkpoint optimizer parameter group is malformed")
        if loaded_group.keys() != current_group.keys():
            raise ValueError("checkpoint optimizer parameter group schema does not match")
        for key, current_value in current_group.items():
            if key != "params" and type(loaded_group[key]) is not type(current_value):
                raise ValueError(
                    f"checkpoint optimizer parameter group field {key!r} has the wrong type"
                )
        for key in ("lr", "eps", "weight_decay", "initial_lr"):
            if key in loaded_group:
                value = _finite_number(
                    loaded_group[key], name=f"optimizer parameter group {key}"
                )
                if value < 0:
                    raise ValueError(
                        f"checkpoint optimizer parameter group {key!r} must be non-negative"
                    )
        betas = loaded_group.get("betas")
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or any(
                not 0 <= _finite_number(beta, name="optimizer beta") < 1
                for beta in betas
            )
        ):
            raise ValueError("checkpoint optimizer betas are malformed")
        loaded_parameters = loaded_group.get("params")
        if not isinstance(loaded_parameters, list) or len(loaded_parameters) != len(
            current_group["params"]
        ):
            raise ValueError("checkpoint optimizer parameters do not match")
        for parameter_id, parameter in zip(
            loaded_parameters, live_group["params"], strict=True
        ):
            if type(parameter_id) is not int:
                raise ValueError("checkpoint optimizer parameter IDs must be integers")
            if parameter_id in parameter_ids:
                raise ValueError("checkpoint optimizer parameter IDs must be unique")
            parameter_ids.add(parameter_id)
            parameter_state = loaded_state.get(parameter_id, {})
            if not isinstance(parameter_state, Mapping):
                raise ValueError("checkpoint optimizer parameter state is malformed")
            expected_state_names = {"step", "exp_avg", "exp_avg_sq"}
            if bool(loaded_group["amsgrad"]):
                expected_state_names.add("max_exp_avg_sq")
            if parameter_state and set(parameter_state) != expected_state_names:
                raise ValueError("checkpoint optimizer parameter state schema does not match")
            for state_name, value in parameter_state.items():
                if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                    raise ValueError("checkpoint optimizer tensor state is malformed")
                if state_name == "step":
                    if value.numel() != 1:
                        raise ValueError("checkpoint optimizer step state is malformed")
                    step = float(value)
                    if step < 0 or not step.is_integer():
                        raise ValueError("checkpoint optimizer step state is malformed")
                elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("checkpoint optimizer tensor state is incompatible")
    if any(type(parameter_id) is not int for parameter_id in loaded_state):
        raise ValueError("checkpoint optimizer state IDs must be integers")
    if set(loaded_state) - parameter_ids:
        raise ValueError("checkpoint optimizer state contains unknown parameters")


def _validate_state_structure(
    current: Mapping[str, Any], loaded: Mapping[str, Any], *, name: str
) -> None:
    if current.keys() != loaded.keys():
        raise ValueError(f"checkpoint {name} state keys do not match")
    for key, current_value in current.items():
        loaded_value = loaded[key]
        if isinstance(current_value, list) and (
            not isinstance(loaded_value, list) or len(current_value) != len(loaded_value)
        ):
            raise ValueError(f"checkpoint {name} state {key!r} is incompatible")


def _finite_rate(value: object, *, name: str) -> float:
    rate = _finite_number(value, name=name)
    if rate < 0:
        raise ValueError(f"{name} must be non-negative")
    return rate


def _validate_scheduler_state(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scheduler_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    *,
    checkpoint_epoch: int,
) -> None:
    _validate_state_structure(scheduler.state_dict(), scheduler_state, name="scheduler")
    scheduler_epoch = scheduler_state.get("last_epoch")
    if type(scheduler_epoch) is not int or scheduler_epoch != checkpoint_epoch:
        raise ValueError("checkpoint scheduler epoch does not match checkpoint history")
    step_count = scheduler_state.get("_step_count")
    if type(step_count) is not int or step_count != checkpoint_epoch + 1:
        raise ValueError("checkpoint scheduler step count does not match checkpoint history")

    loaded_groups = optimizer_state.get("param_groups")
    base_lrs = scheduler_state.get("base_lrs")
    last_lrs = scheduler_state.get("_last_lr")
    target_base_lrs = scheduler.state_dict().get("base_lrs")
    if not all(isinstance(values, list) for values in (loaded_groups, base_lrs, last_lrs)):
        raise ValueError("checkpoint scheduler learning-rate state is malformed")
    if not isinstance(target_base_lrs, list) or not (
        len(loaded_groups) == len(base_lrs) == len(last_lrs) == len(target_base_lrs)
    ):
        raise ValueError("checkpoint scheduler learning-rate groups do not match")
    for index, (group, base_lr, last_lr, target_base_lr) in enumerate(
        zip(loaded_groups, base_lrs, last_lrs, target_base_lrs, strict=True)
    ):
        if not isinstance(group, Mapping):
            raise ValueError("checkpoint optimizer parameter group is malformed")
        base = _finite_rate(base_lr, name=f"scheduler base LR {index}")
        last = _finite_rate(last_lr, name=f"scheduler last LR {index}")
        target_base = _finite_rate(
            target_base_lr, name=f"expected scheduler base LR {index}"
        )
        optimizer_base = _finite_rate(
            group.get("initial_lr"), name=f"optimizer initial LR {index}"
        )
        optimizer_last = _finite_rate(group.get("lr"), name=f"optimizer LR {index}")
        if base != target_base or base != optimizer_base:
            raise ValueError("checkpoint scheduler base learning rate does not match")
        if last != optimizer_last:
            raise ValueError("checkpoint optimizer and scheduler learning rates do not match")


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
    _validated_metadata(
        payload,
        expected_model_alias=model_alias,
        expected_resolved_config=config_copy,
        expected_selection_metric=selection_metric,
    )
    _validate_model_state(model, payload["model_state"])
    _validate_optimizer_state(optimizer, payload["optimizer_state"])
    _validate_scheduler_state(
        scheduler,
        payload["scheduler_state"],
        payload["optimizer_state"],
        checkpoint_epoch=epoch,
    )
    _validate_state_structure(scaler.state_dict(), payload["scaler_state"], name="scaler")
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
    expected_resolved_config: Mapping[str, Any] | None = None,
    expected_selection_metric: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
) -> dict[str, Any]:
    """Validate identity, restore requested state, and return resume metadata."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    metadata = _validated_metadata(
        checkpoint,
        expected_model_alias=expected_model_alias,
        expected_resolved_config=expected_resolved_config,
        expected_selection_metric=expected_selection_metric,
    )
    model_state = _require_state_mapping(checkpoint, "model_state")
    optimizer_state = _require_state_mapping(checkpoint, "optimizer_state")
    scheduler_state = _require_state_mapping(checkpoint, "scheduler_state")
    scaler_state = _require_state_mapping(checkpoint, "scaler_state")
    _validate_model_state(model, model_state)
    if optimizer is not None:
        _validate_optimizer_state(optimizer, optimizer_state)
    if scheduler is not None:
        _validate_scheduler_state(
            scheduler,
            scheduler_state,
            optimizer_state,
            checkpoint_epoch=metadata["epoch"],
        )
    if scaler is not None:
        _validate_state_structure(scaler.state_dict(), scaler_state, name="scaler")
    if early_stopping is not None:
        if (
            early_stopping.patience != metadata["early_stopping_state"]["patience"]
            or early_stopping.min_delta != metadata["early_stopping_state"]["min_delta"]
        ):
            raise ValueError("early-stopping policy does not match the checkpoint")

    originals = {
        "model": deepcopy(model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "scaler": deepcopy(scaler.state_dict()) if scaler is not None else None,
        "early_stopping": (
            deepcopy(early_stopping.state_dict()) if early_stopping is not None else None
        ),
    }
    try:
        model.load_state_dict(model_state)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
        if early_stopping is not None:
            early_stopping.load_state_dict(metadata["early_stopping_state"])
    except Exception as error:
        model.load_state_dict(originals["model"])
        if optimizer is not None:
            optimizer.load_state_dict(originals["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(originals["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(originals["scaler"])
        if early_stopping is not None:
            early_stopping.load_state_dict(originals["early_stopping"])
        raise ValueError("checkpoint state is incompatible with the supplied objects") from error
    return metadata
