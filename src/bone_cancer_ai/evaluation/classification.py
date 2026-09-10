"""Binary classification metrics for baseline and final evaluation."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 10
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, boundaries[1:-1]), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = assignments == bin_index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(targets[selected].mean())
            )
    return float(error)


def binary_classification_metrics(
    logits: object, targets: object, threshold: float = 0.5
) -> dict[str, object]:
    """Compute binary metrics from one-dimensional logits at a fixed threshold."""
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("threshold must be a finite probability between 0 and 1")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability between 0 and 1")

    try:
        logit_tensor = torch.as_tensor(logits).detach()
        target_tensor = torch.as_tensor(targets).detach()
    except (TypeError, ValueError) as error:
        raise ValueError("logits and targets must be numeric array-like values") from error
    if logit_tensor.ndim != 1 or target_tensor.ndim != 1:
        raise ValueError("logits and targets must be one-dimensional")
    if logit_tensor.shape != target_tensor.shape:
        raise ValueError("logits and targets must have the same shape")
    if logit_tensor.numel() == 0:
        raise ValueError("logits and targets must be non-empty")
    if logit_tensor.is_complex() or target_tensor.is_complex():
        raise ValueError("logits and targets must be real-valued")
    if not bool(torch.isfinite(logit_tensor).all()) or not bool(
        torch.isfinite(target_tensor).all()
    ):
        raise ValueError("logits and targets must contain only finite values")
    if not bool(((target_tensor == 0) | (target_tensor == 1)).all()):
        raise ValueError("targets must contain only binary labels 0 and 1")

    logit_array = logit_tensor.cpu().to(torch.float64).numpy()
    target_array = target_tensor.cpu().numpy().astype(np.int64)
    probabilities = torch.sigmoid(torch.from_numpy(logit_array)).numpy()
    predictions = (probabilities >= threshold).astype(np.int64)

    negative = target_array == 0
    positive = target_array == 1
    predicted_negative = predictions == 0
    predicted_positive = predictions == 1
    tn = int(np.sum(negative & predicted_negative))
    fp = int(np.sum(negative & predicted_positive))
    fn = int(np.sum(positive & predicted_negative))
    tp = int(np.sum(positive & predicted_positive))

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (
        float((tp * tn - fp * fn) / np.sqrt(mcc_denominator))
        if mcc_denominator
        else None
    )
    balanced_accuracy = (
        float((recall + specificity) / 2.0)
        if recall is not None and specificity is not None
        else None
    )
    has_positive = bool(positive.any())
    has_negative = bool(negative.any())

    return {
        "AUROC": (
            float(roc_auc_score(target_array, probabilities))
            if has_positive and has_negative
            else None
        ),
        "AUPRC": (
            float(average_precision_score(target_array, probabilities)) if has_positive else None
        ),
        "accuracy": float((tp + tn) / target_array.size),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "F1": f1,
        "MCC": mcc,
        "Brier": float(np.mean(np.square(probabilities - target_array))),
        "ECE": _expected_calibration_error(probabilities, target_array),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
