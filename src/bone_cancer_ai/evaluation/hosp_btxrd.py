"""Subtype-aware evaluation for the BTXRD migration run.

This module is deliberately separate from the frozen binary Phase 9 evaluator.
It only consumes predictions from the BTXRD calibration and test partitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator


def _probability_matrix(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty [N, C] matrix with C >= 2")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = result.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")
    return result


def _multiclass_targets(
    values: object, num_classes: int, expected_size: int | None = None
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("targets must be a matching non-empty one-dimensional array")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("targets must be numeric class indices") from error
    labels = numeric.astype(np.int64)
    if (
        labels.ndim != 1
        or labels.size == 0
        or (expected_size is not None and labels.size != expected_size)
        or not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError("targets must be a matching non-empty one-dimensional array")
    if ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError("targets must be valid class indices")
    return labels


@dataclass(frozen=True)
class MulticlassConformalCalibrator:
    """Split-conformal prediction sets calibrated on held-out subtype rows."""

    alpha: float
    qhat: float
    calibration_count: int
    num_classes: int

    @classmethod
    def fit(
        cls, probabilities: object, targets: object, *, alpha: float = 0.10
    ) -> MulticlassConformalCalibrator:
        values = _probability_matrix(probabilities)
        labels = _multiclass_targets(targets, values.shape[1], values.shape[0])
        if isinstance(alpha, bool) or not isinstance(alpha, int | float):
            raise ValueError("alpha must be between 0 and 1")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        scores = 1.0 - values[np.arange(labels.size), labels]
        rank = min(scores.size, math.ceil((scores.size + 1) * (1.0 - alpha)))
        qhat = float(np.sort(scores)[rank - 1])
        return cls(
            alpha=alpha,
            qhat=qhat,
            calibration_count=int(labels.size),
            num_classes=int(values.shape[1]),
        )

    def predict_sets(self, probabilities: object) -> list[list[int]]:
        values = _probability_matrix(probabilities)
        if values.shape[1] != self.num_classes:
            raise ValueError("probability class count does not match the calibrator")
        included = 1.0 - values <= self.qhat + 1e-12
        return [np.flatnonzero(row).astype(int).tolist() for row in included]

    def summarize(self, probabilities: object, targets: object) -> dict[str, float | int]:
        values = _probability_matrix(probabilities)
        labels = _multiclass_targets(targets, values.shape[1], values.shape[0])
        sets = self.predict_sets(values)
        sizes = np.asarray([len(item) for item in sets], dtype=np.int64)
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

    def state_dict(self) -> dict[str, float | int]:
        return {
            "alpha": self.alpha,
            "qhat": self.qhat,
            "calibration_count": self.calibration_count,
            "num_classes": self.num_classes,
        }

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready calibration record."""
        return self.state_dict()

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MulticlassConformalCalibrator:
        required = {"alpha", "qhat", "calibration_count", "num_classes"}
        if set(state) != required:
            raise ValueError("invalid multiclass conformal state")
        return cls(
            alpha=float(state["alpha"]),
            qhat=float(state["qhat"]),
            calibration_count=int(state["calibration_count"]),
            num_classes=int(state["num_classes"]),
        )


def subtype_classification_metrics(logits: object, targets: object) -> dict[str, object]:
    """Compute subtype metrics for validated, subtype-labeled rows only."""

    values = torch.as_tensor(logits).detach()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must be a non-empty [N, C] matrix with C >= 2")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("logits must be finite")
    labels = _multiclass_targets(targets, int(values.shape[1]), int(values.shape[0]))
    probabilities = torch.softmax(values.to(torch.float64), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    present = np.unique(labels)
    result: dict[str, object] = {
        "sample_count": int(labels.size),
        "class_count": int(values.shape[1]),
        "classes_present": present.astype(int).tolist(),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=list(range(values.shape[1])))),
    }
    if present.size == values.shape[1]:
        result["AUROC"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    else:
        result["AUROC"] = None
    return result


def evaluate_hosp_btxrd_predictions(
    *,
    calibration_binary_logits: object,
    calibration_binary_targets: object,
    test_binary_logits: object,
    test_binary_targets: object,
    calibration_subtype_probabilities: object | None = None,
    calibration_subtype_targets: object | None = None,
    test_subtype_logits: object | None = None,
    test_subtype_targets: object | None = None,
    alpha_binary: float = 0.10,
    alpha_subtype: float = 0.10,
) -> dict[str, object]:
    """Evaluate BTXRD calibration/test predictions without importing old metrics."""

    binary = binary_classification_metrics(test_binary_logits, test_binary_targets)
    cal_binary_probabilities = (
        torch.sigmoid(torch.as_tensor(calibration_binary_logits)).cpu().numpy()
    )
    test_binary_probabilities = torch.sigmoid(torch.as_tensor(test_binary_logits)).cpu().numpy()
    binary_calibrator = BinaryConformalCalibrator.fit(
        cal_binary_probabilities, calibration_binary_targets, alpha=alpha_binary
    )
    result: dict[str, object] = {
        "protocol": "btxrd_fresh_calibration_and_test",
        "binary": binary,
        "binary_conformal": binary_calibrator.summarize(
            test_binary_probabilities, test_binary_targets
        ),
    }
    if any(
        value is None
        for value in (
            calibration_subtype_probabilities,
            calibration_subtype_targets,
            test_subtype_logits,
            test_subtype_targets,
        )
    ):
        result["subtype"] = {"status": "unavailable"}
        return result
    calibration_probabilities = _probability_matrix(calibration_subtype_probabilities)
    calibration_targets = _multiclass_targets(
        calibration_subtype_targets,
        calibration_probabilities.shape[1],
        calibration_probabilities.shape[0],
    )
    test_logits = torch.as_tensor(test_subtype_logits).detach()
    test_probabilities = torch.softmax(test_logits.to(torch.float64), dim=1).cpu().numpy()
    test_targets = _multiclass_targets(
        test_subtype_targets, test_probabilities.shape[1], test_probabilities.shape[0]
    )
    subtype_calibrator = MulticlassConformalCalibrator.fit(
        calibration_probabilities, calibration_targets, alpha=alpha_subtype
    )
    result["subtype"] = subtype_classification_metrics(test_logits, test_targets)
    result["subtype_conformal"] = subtype_calibrator.summarize(test_probabilities, test_targets)
    result["subtype_calibrator"] = subtype_calibrator.state_dict()
    return result


@torch.inference_mode()
def predict_hosp_multitask(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    amp: bool = False,
) -> dict[str, Any]:
    """Collect masked binary/subtype logits from one split loader."""

    binary_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []
    rejected = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=bool(amp and device.type == "cuda")
        ):
            outputs = model(images)
        if not isinstance(outputs, dict):
            outputs = dict(outputs)
        binary_logits = outputs["binary_logits"].detach().reshape(-1).cpu()
        subtype_logits = outputs.get("subtype_logits")
        masks = batch["valid_task_mask"]
        sample_mask = masks["sample"].bool()
        binary_mask = (sample_mask & masks["binary"].bool()).cpu()
        subtype_mask = (sample_mask & masks["subtype"].bool()).cpu()
        metadata = batch["metadata"]
        sample_ids = (
            metadata["sample_id"]
            if isinstance(metadata["sample_id"], list)
            else list(metadata["sample_id"])
        )
        for index, sample_id in enumerate(sample_ids):
            if bool(binary_mask[index]):
                binary_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "target": int(batch["binary_label"][index]),
                        "logit": float(binary_logits[index]),
                        "probability": float(torch.sigmoid(binary_logits[index])),
                    }
                )
            else:
                rejected += 1
            if subtype_logits is not None and bool(subtype_mask[index]):
                logits = subtype_logits[index].detach().cpu()
                subtype_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "target": int(batch["subtype_label"][index]),
                        "logits": logits.tolist(),
                        "probabilities": torch.softmax(logits, dim=0).tolist(),
                        "prediction": int(logits.argmax()),
                    }
                )
    return {
        "binary": pd.DataFrame(binary_rows),
        "subtype": pd.DataFrame(subtype_rows),
        "rejected_count": rejected,
    }


__all__ = [
    "MulticlassConformalCalibrator",
    "evaluate_hosp_btxrd_predictions",
    "predict_hosp_multitask",
    "subtype_classification_metrics",
]
