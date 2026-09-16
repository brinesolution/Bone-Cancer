"""Pure contracts for validation-only Phase 8 experiments."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import torch
from torch import nn

from bone_cancer_ai.optimization.phase8 import (
    ensure_validation_only,
    load_compatible_parent_state,
    phase8_fingerprint,
    select_validation_candidate,
    stratified_subset,
    validation_rank,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(10)],
            "binary_label": [0] * 6 + [1] * 4,
            "assigned_split": ["fit_train"] * 10,
        }
    )


def test_stratified_subset_is_deterministic_balanced_and_order_independent() -> None:
    frame = _frame()

    first = stratified_subset(frame, limit=6, seed=42)
    second = stratified_subset(frame.sample(frac=1.0, random_state=7), limit=6, seed=42)

    assert first["sample_id"].tolist() == second["sample_id"].tolist()
    assert len(first) == 6
    assert set(first["binary_label"]) == {0, 1}
    assert first["sample_id"].is_unique


@pytest.mark.parametrize("limit", [0, 1, 11, True])
def test_stratified_subset_rejects_unsafe_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        stratified_subset(_frame(), limit=limit, seed=42)


def test_validation_only_guard_rejects_nested_test_references() -> None:
    ensure_validation_only({"splits": ["fit_train", "val"], "selection_split": "val"})

    with pytest.raises(ValueError, match="test"):
        ensure_validation_only(
            {"splits": ["fit_train", "val"], "output": "artifacts/test/predictions.csv"}
        )


def test_validation_rank_uses_auroc_then_auprc_then_balanced_accuracy() -> None:
    lower_auroc = {
        "AUROC": 0.89,
        "AUPRC": 0.99,
        "balanced_accuracy": 0.99,
        "sensitivity": 0.99,
        "F1": 0.99,
        "ECE": 0.01,
        "Brier": 0.01,
    }
    better_auroc = dict(lower_auroc, AUROC=0.90, AUPRC=0.80)
    better_auprc = dict(better_auroc, AUPRC=0.81, balanced_accuracy=0.70)

    assert validation_rank(better_auroc) > validation_rank(lower_auroc)
    assert validation_rank(better_auprc) > validation_rank(better_auroc)


def test_candidate_selection_is_validation_only_and_deterministic() -> None:
    base = {
        "AUROC": 0.90,
        "AUPRC": 0.91,
        "balanced_accuracy": 0.80,
        "sensitivity": 0.85,
        "F1": 0.82,
        "ECE": 0.05,
        "Brier": 0.10,
    }
    rows = [
        {"candidate": "later", "validation": deepcopy(base)},
        {"candidate": "alpha", "validation": deepcopy(base)},
        {"candidate": "worse", "validation": dict(base, AUROC=0.89)},
    ]

    assert select_validation_candidate(rows)["candidate"] == "alpha"


def test_compatible_parent_transfer_requires_every_target_tensor() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    parent = {name: value.clone() + 1 for name, value in model.state_dict().items()}
    parent["unused.weight"] = torch.ones(1)

    report = load_compatible_parent_state(model, parent)

    assert report["loaded"] == len(model.state_dict())
    assert report["unused_parent"] == ["unused.weight"]
    assert all(torch.equal(model.state_dict()[name], parent[name]) for name in model.state_dict())

    parent.pop("0.bias")
    with pytest.raises(ValueError, match="missing"):
        load_compatible_parent_state(model, parent)


def test_phase8_fingerprint_is_order_independent_and_rejects_non_json_values() -> None:
    left = phase8_fingerprint({"b": 2, "a": [1, 3]})
    right = phase8_fingerprint({"a": [1, 3], "b": 2})

    assert left == right
    assert len(left) == 64
    with pytest.raises(ValueError, match="JSON"):
        phase8_fingerprint({"bad": {1, 2}})
