"""Deterministic selection of representative outcome and uncertainty cases."""

from __future__ import annotations

import pandas as pd


def select_gallery_cases(predictions: pd.DataFrame, *, per_category: int = 4) -> pd.DataFrame:
    """Select confident outcomes/failures and the most uncertain abstentions."""
    required = {
        "sample_id",
        "target",
        "prediction",
        "probability",
        "uncertainty",
        "conformal_size",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"gallery predictions missing columns: {sorted(missing)}")
    if type(per_category) is not int or per_category <= 0:
        raise ValueError("per_category must be a positive integer")
    frame = predictions.copy()
    frame["confidence"] = (frame["probability"] - 0.5).abs() * 2.0
    outcomes = {
        "true_positive": (frame["target"] == 1) & (frame["prediction"] == 1),
        "true_negative": (frame["target"] == 0) & (frame["prediction"] == 0),
        "false_positive": (frame["target"] == 0) & (frame["prediction"] == 1),
        "false_negative": (frame["target"] == 1) & (frame["prediction"] == 0),
    }
    selected = []
    for category, mask in outcomes.items():
        rows = frame.loc[mask].sort_values(
            ["confidence", "sample_id"], ascending=[False, True], kind="stable"
        )
        if not rows.empty:
            rows = rows.head(per_category).copy()
            rows.insert(0, "category", category)
            selected.append(rows)
    rejected = frame.loc[frame["conformal_size"] != 1]
    if rejected.empty:
        rejected = frame
    rejected = rejected.sort_values(
        ["uncertainty", "sample_id"], ascending=[False, True], kind="stable"
    ).head(per_category)
    if not rejected.empty:
        rejected = rejected.copy()
        rejected.insert(0, "category", "uncertain_rejected")
        selected.append(rejected)
    if not selected:
        raise ValueError("no gallery cases are available")
    return pd.concat(selected, ignore_index=True)


__all__ = ["select_gallery_cases"]
