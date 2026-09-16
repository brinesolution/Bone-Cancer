"""Strict reusable primitives for Phase 8 validation experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from torch import nn

_TEST_TOKEN = re.compile(r"(?:^|[\\/_.-])test(?:$|[\\/_.-])", re.IGNORECASE)
_RANK_FIELDS = ("AUROC", "AUPRC", "balanced_accuracy", "sensitivity", "F1")


def ensure_validation_only(value: object) -> None:
    """Reject configuration values that reference the held-out test split."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            ensure_validation_only(key)
            ensure_validation_only(nested)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for nested in value:
            ensure_validation_only(nested)
    elif isinstance(value, str) and _TEST_TOKEN.search(value):
        raise ValueError(f"Phase 8 must not reference test data: {value!r}")


def _finite_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool):
        raise ValueError(f"validation metric {name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"validation metric {name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"validation metric {name} must be finite")
    return result


def validation_rank(metrics: Mapping[str, object]) -> tuple[float, ...]:
    """Return the preregistered validation-only lexicographic ranking key."""

    positive = tuple(_finite_metric(metrics, name) for name in _RANK_FIELDS)
    return (*positive, -_finite_metric(metrics, "ECE"), -_finite_metric(metrics, "Brier"))


def select_validation_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by validation rank, breaking exact ties by candidate name."""

    if not candidates:
        raise ValueError("at least one validation candidate is required")
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        name = candidate.get("candidate")
        metrics = candidate.get("validation")
        if not isinstance(name, str) or not name or not isinstance(metrics, Mapping):
            raise ValueError("candidate and validation metrics are required")
        normalized.append(dict(candidate))
    normalized.sort(key=lambda row: row["candidate"])
    return max(normalized, key=lambda row: validation_rank(row["validation"]))


def _sample_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def stratified_subset(frame: pd.DataFrame, *, limit: int, seed: int) -> pd.DataFrame:
    """Choose a stable proportional binary subset without depending on row order."""

    if type(limit) is not int or limit < 2 or limit > len(frame):
        raise ValueError("limit must be an integer from 2 through the frame length")
    required = {"sample_id", "binary_label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"subset frame is missing columns: {sorted(missing)}")
    labels = pd.to_numeric(frame["binary_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("subset binary_label must contain only 0 and 1")
    if labels.nunique() != 2 or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("subset frame requires both classes and unique sample_id values")

    working = frame.copy()
    working["binary_label"] = labels.astype(int)
    working["_rank"] = working["sample_id"].astype(str).map(lambda item: _sample_rank(item, seed))
    class_counts = working["binary_label"].value_counts().sort_index()
    negative = max(1, min(int(class_counts[0]), round(limit * class_counts[0] / len(working))))
    positive = limit - negative
    if positive < 1 or positive > int(class_counts[1]):
        positive = max(1, min(int(class_counts[1]), positive))
        negative = limit - positive
    if negative < 1 or negative > int(class_counts[0]) or negative + positive != limit:
        raise ValueError("limit cannot preserve both binary classes")
    selected = pd.concat(
        [
            working.loc[working["binary_label"].eq(0)]
            .sort_values("_rank")
            .head(negative),
            working.loc[working["binary_label"].eq(1)]
            .sort_values("_rank")
            .head(positive),
        ],
        ignore_index=True,
    )
    return selected.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)


def load_compatible_parent_state(
    model: nn.Module, parent_state: Mapping[str, object]
) -> dict[str, Any]:
    """Load every target tensor while allowing unused parent-only branch tensors."""

    target = model.state_dict()
    missing = sorted(set(target) - set(parent_state))
    if missing:
        raise ValueError(f"parent model state is missing target tensors: {missing}")
    selected: dict[str, torch.Tensor] = {}
    for name, current in target.items():
        loaded = parent_state[name]
        if not isinstance(loaded, torch.Tensor):
            raise ValueError(f"parent model tensor {name!r} is not a tensor")
        if current.shape != loaded.shape or current.dtype != loaded.dtype:
            raise ValueError(f"parent model tensor {name!r} is incompatible")
        if not bool(torch.isfinite(loaded).all()):
            raise ValueError(f"parent model tensor {name!r} is non-finite")
        selected[name] = loaded
    model.load_state_dict(selected, strict=True)
    return {
        "loaded": len(selected),
        "unused_parent": sorted(set(parent_state) - set(target)),
    }


def phase8_fingerprint(value: Mapping[str, object]) -> str:
    """Hash strict JSON configuration data canonically."""

    try:
        encoded = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("fingerprint input must be strict JSON data") from error
    return hashlib.sha256(encoded).hexdigest()
