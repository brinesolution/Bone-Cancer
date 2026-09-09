"""Grouped calibration splitting and optional class-balanced PyTorch sampling."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import WeightedRandomSampler


def split_generation_id(frame: pd.DataFrame) -> str:
    """Fingerprint every persisted field in a split bundle except the fingerprint itself."""
    required = {"record_id", "sample_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split fingerprint missing columns: {sorted(missing)}")
    columns = sorted(column for column in frame.columns if column != "split_generation_id")
    stable = frame.loc[:, columns].copy().astype("string").fillna("")
    stable = stable.sort_values(["record_id", "sample_id"])
    payload = stable.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_score(
    train_labels: pd.Series,
    calibration_labels: pd.Series,
    fraction: float,
) -> float:
    score = abs(len(calibration_labels) / len(train_labels) - fraction)
    for label, total in train_labels.value_counts().items():
        actual = int(calibration_labels.eq(label).sum()) / int(total)
        score += abs(actual - fraction)
        fit_count = int(total) - int(calibration_labels.eq(label).sum())
        if int(calibration_labels.eq(label).sum()) == 0 or fit_count == 0:
            score += 100.0
    return score


def assign_internal_splits(
    clean: pd.DataFrame,
    calibration_fraction: float,
    *,
    seed: int,
    search_trials: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split only the official-training pool while keeping connected groups intact."""
    required = {
        "record_id",
        "official_split",
        "clean_pool",
        "assigned_split",
        "split_group_id",
        "binary_label",
        "is_eligible",
    }
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Clean manifest missing columns: {sorted(missing)}")
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    if search_trials < 1:
        raise ValueError("search_trials must be positive")
    result = clean.sort_values("record_id").reset_index(drop=True).copy()
    pool_mask = result["is_eligible"].astype(bool) & result["clean_pool"].eq("train_pool")
    pool = result.loc[pool_mask].copy()
    if pool["split_group_id"].nunique() < 2:
        raise ValueError("At least two eligible training groups are required for calibration")
    labels = pd.to_numeric(pool["binary_label"], errors="coerce")
    if labels.isna().any():
        raise ValueError("Training pool contains a missing/non-numeric binary label")
    groups = pool["split_group_id"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=search_trials,
        test_size=calibration_fraction,
        random_state=seed,
    )
    best: tuple[float, str, set[str]] | None = None
    for _, calibration_indices in splitter.split(np.zeros(len(pool)), labels, groups):
        calibration_groups = set(groups.iloc[calibration_indices])
        calibration_labels = labels.iloc[calibration_indices]
        score = _candidate_score(labels, calibration_labels, calibration_fraction)
        digest = hashlib.sha256("|".join(sorted(calibration_groups)).encode()).hexdigest()
        candidate = (score, digest, calibration_groups)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No calibration split candidate was generated")
    selected_groups = best[2]
    result.loc[pool_mask, "assigned_split"] = np.where(
        result.loc[pool_mask, "split_group_id"].astype(str).isin(selected_groups),
        "calibration",
        "fit_train",
    )
    eligible = result[result["is_eligible"].astype(bool)]
    generation_id = split_generation_id(eligible)
    result["split_generation_id"] = generation_id
    assigned_counts = {
        str(key): int(value) for key, value in eligible["assigned_split"].value_counts().items()
    }
    class_counts: dict[str, dict[str, int]] = {}
    for (split, label), count in eligible.groupby(
        ["assigned_split", "binary_label"], sort=True
    ).size().items():
        class_counts.setdefault(str(split), {})[str(int(label))] = int(count)
    calibration_count = assigned_counts.get("calibration", 0)
    summary = {
        "calibration_fraction_target": calibration_fraction,
        "actual_calibration_fraction": calibration_count / len(pool),
        "search_trials": search_trials,
        "candidate_score": best[0],
        "training_pool_records": len(pool),
        "training_pool_groups": int(groups.nunique()),
        "calibration_groups": len(selected_groups),
        "assigned_split_counts": assigned_counts,
        "class_counts_by_split": class_counts,
        "heldout_labels_used_for_calibration_selection": False,
        "split_generation_id": generation_id,
    }
    return result, summary


def validate_final_splits(assigned: pd.DataFrame) -> dict[str, Any]:
    """Validate record/group disjointness across eligible final partitions."""
    required = {
        "record_id",
        "sample_id",
        "split_group_id",
        "assigned_split",
        "split_generation_id",
        "is_eligible",
    }
    missing = required - set(assigned.columns)
    if missing:
        raise ValueError(f"Assigned manifest missing columns: {sorted(missing)}")
    eligible = assigned[assigned["is_eligible"].astype(bool)]
    allowed = {"fit_train", "calibration", "val", "test"}
    unassigned = int((~eligible["assigned_split"].isin(allowed)).sum())
    group_overlap = int(
        (eligible.groupby("split_group_id")["assigned_split"].nunique() > 1).sum()
    )
    record_overlap = int(
        (eligible.groupby("record_id")["assigned_split"].nunique() > 1).sum()
        + eligible["record_id"].duplicated().sum()
    )
    sample_overlap = int(
        (eligible.groupby("sample_id")["assigned_split"].nunique() > 1).sum()
        + eligible["sample_id"].duplicated().sum()
    )
    generation_ids = eligible["split_generation_id"].dropna().astype(str).unique()
    generation_mismatch = int(
        len(generation_ids) != 1
        or (len(generation_ids) == 1 and generation_ids[0] != split_generation_id(eligible))
    )
    heldout_to_training = int(
        (
            eligible["official_split"].isin(["val", "test"])
            & eligible["assigned_split"].isin(["fit_train", "calibration"])
        ).sum()
    )
    return {
        "eligible_records": len(eligible),
        "group_overlap": group_overlap,
        "record_overlap": record_overlap,
        "sample_overlap": sample_overlap,
        "generation_mismatch": generation_mismatch,
        "unassigned_eligible_records": unassigned,
        "official_heldout_records_assigned_to_training": heldout_to_training,
    }


def binary_class_weights(frame: pd.DataFrame) -> torch.Tensor:
    """Return balanced inverse-frequency weights ordered by integer class label."""
    labels = pd.to_numeric(frame["binary_label"], errors="coerce")
    if labels.isna().any() or labels.empty:
        raise ValueError("Binary class weights require nonempty numeric labels")
    counts = labels.astype(int).value_counts().sort_index()
    expected = list(range(int(counts.index.max()) + 1))
    if list(counts.index) != expected:
        raise ValueError("Binary labels must be contiguous from zero")
    total = len(labels)
    return torch.tensor(
        [total / (len(counts) * int(counts[label])) for label in expected],
        dtype=torch.float32,
    )


def make_weighted_sampler(frame: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    """Build a seeded inverse-frequency sample stream without changing the manifest."""
    labels = pd.to_numeric(frame["binary_label"], errors="raise").astype(int)
    weights = binary_class_weights(frame)
    sample_weights = torch.tensor([float(weights[label]) for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
