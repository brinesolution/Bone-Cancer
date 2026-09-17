"""Prediction validation and final binary-model comparison figures."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .calibration import binary_calibration_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREDICTION_COLUMNS = ["sample_id", "target", "logit", "probability", "prediction"]


def validate_prediction_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Validate and sample-sort one sealed binary prediction table."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} predictions must be a nonempty DataFrame")
    missing = set(PREDICTION_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"{label} predictions missing columns: {sorted(missing)}")
    result = frame.loc[:, PREDICTION_COLUMNS].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].eq("").any() or result["sample_id"].duplicated().any():
        raise ValueError(f"{label} sample IDs must be nonempty and unique")
    for column in ("target", "logit", "probability", "prediction"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[["target", "logit", "probability", "prediction"]]).all().all():
        raise ValueError(f"{label} predictions contain non-finite values")
    if not result["target"].isin([0, 1]).all() or not result["prediction"].isin([0, 1]).all():
        raise ValueError(f"{label} targets and predictions must be binary")
    expected_probability = 1.0 / (1.0 + np.exp(-result["logit"].to_numpy(dtype=float)))
    if not np.allclose(
        result["probability"].to_numpy(dtype=float), expected_probability, rtol=1e-6, atol=1e-8
    ):
        raise ValueError(f"{label} probability does not match its logit")
    expected_prediction = (result["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
    if not np.array_equal(result["prediction"].to_numpy(dtype=int), expected_prediction):
        raise ValueError(f"{label} fixed-threshold predictions are inconsistent")
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def ensure_matching_cohort(predictions: Mapping[str, pd.DataFrame]) -> None:
    """Require every model to cover identical ordered sample IDs and targets."""
    if not predictions:
        raise ValueError("at least one prediction table is required")
    reference_name, reference = next(iter(predictions.items()))
    reference = validate_prediction_frame(reference, label=reference_name)
    for name, frame in predictions.items():
        checked = validate_prediction_frame(frame, label=name)
        if not checked["sample_id"].equals(reference["sample_id"]):
            raise ValueError(f"{name} sample IDs differ from {reference_name}")
        if not checked["target"].astype(int).equals(reference["target"].astype(int)):
            raise ValueError(f"{name} targets differ from {reference_name}")


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.png")
    try:
        figure.savefig(temporary, dpi=160, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return destination


def _curve_figure(
    predictions: Mapping[str, pd.DataFrame], *, curve: str, destination: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, frame in predictions.items():
        target = frame["target"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)
        if curve == "roc":
            x, y, _ = roc_curve(target, probability)
            axis.set(
                xlabel="False-positive rate",
                ylabel="True-positive rate",
                title="Test ROC curves",
            )
        else:
            y, x, _ = precision_recall_curve(target, probability)
            axis.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curves")
        axis.plot(x, y, linewidth=1.8, label=name)
    if curve == "roc":
        axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return _save_figure(figure, destination)


def _calibration_figure(predictions: Mapping[str, pd.DataFrame], destination: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Ideal")
    for name, frame in predictions.items():
        curve = binary_calibration_curve(frame["probability"], frame["target"], bins=10)
        axis.plot(
            curve["mean_probability"],
            curve["observed_frequency"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=name,
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed tumour frequency",
        title="Test calibration curves",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return _save_figure(figure, destination)


def _confusion_figure(name: str, frame: pd.DataFrame, destination: Path) -> Path:
    target = frame["target"].to_numpy(dtype=int)
    prediction = frame["prediction"].to_numpy(dtype=int)
    matrix = np.asarray(
        [
            [np.sum((target == 0) & (prediction == 0)), np.sum((target == 0) & (prediction == 1))],
            [np.sum((target == 1) & (prediction == 0)), np.sum((target == 1) & (prediction == 1))],
        ],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["normal", "bone_tumour"])
    axis.set_yticks([0, 1], ["normal", "bone_tumour"])
    axis.set(xlabel="Predicted", ylabel="Actual", title=name)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save_figure(figure, destination)


def plot_final_comparison(
    predictions: Mapping[str, pd.DataFrame], output_root: Path
) -> dict[str, object]:
    """Create required ROC, PR, calibration and confusion-matrix figures."""
    checked = {
        name: validate_prediction_frame(frame, label=name) for name, frame in predictions.items()
    }
    ensure_matching_cohort(checked)
    root = Path(output_root)
    confusion_root = root / "confusion_matrices"
    confusion = [
        _confusion_figure(name, frame, confusion_root / f"{name}.png")
        for name, frame in checked.items()
    ]
    return {
        "roc": _curve_figure(checked, curve="roc", destination=root / "roc_curves.png"),
        "pr": _curve_figure(checked, curve="pr", destination=root / "pr_curves.png"),
        "calibration": _calibration_figure(checked, root / "calibration_curves.png"),
        "confusion_matrices": confusion,
    }


__all__ = [
    "PREDICTION_COLUMNS",
    "ensure_matching_cohort",
    "plot_final_comparison",
    "validate_prediction_frame",
]
