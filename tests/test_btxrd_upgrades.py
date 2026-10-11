from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bone_cancer_ai.evaluation.calibration import TemperatureScaler
from bone_cancer_ai.evaluation.hosp_btxrd import subtype_classification_metrics
from bone_cancer_ai.evaluation.robustness import build_robustness_report
from bone_cancer_ai.inference.quality import assess_image_quality
from bone_cancer_ai.inference.report import btxrd_result_to_html
from bone_cancer_ai.losses.classification import masked_cross_entropy


def test_subtype_metrics_include_named_per_class_confusion_details() -> None:
    result = subtype_classification_metrics(
        torch.tensor([[4.0, 0.0], [3.0, 1.0], [0.0, 4.0], [0.0, 2.0]]),
        torch.tensor([0, 1, 1, 1]),
        class_names=("alpha", "beta"),
    )

    assert result["confusion_matrix"] == [[1, 0], [1, 2]]
    assert result["per_class"][0] == {
        "index": 0,
        "name": "alpha",
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(2 / 3),
        "support": 1,
    }


def test_temperature_scaler_is_fit_only_on_supplied_calibration_logits() -> None:
    logits = torch.tensor([[5.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 5.0]])
    targets = torch.tensor([1, 1, 0, 0])
    scaler = TemperatureScaler.fit(logits, targets)

    assert scaler.temperature > 0.0
    assert scaler.nll(logits, targets) <= scaler.nll(logits, targets, temperature=1.0)
    assert scaler.to_dict()["temperature"] == pytest.approx(scaler.temperature)


def test_masked_cross_entropy_accepts_explicit_subtype_class_weights() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    mask = torch.tensor([True, True])

    loss = masked_cross_entropy(
        logits, targets, mask, class_weights=torch.tensor([1.0, 3.0])
    )
    expected = torch.nn.functional.cross_entropy(
        logits, targets, weight=torch.tensor([1.0, 3.0])
    )
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))


def test_image_quality_reports_reviewable_low_contrast_input() -> None:
    report = assess_image_quality(np.full((80, 96, 3), 128, dtype=np.uint8))

    assert report.status == "review"
    assert "low_contrast" in report.warnings
    assert report.to_dict()["height"] == 80


def test_html_report_contains_prediction_and_provenance() -> None:
    class Result:
        subtype_prediction = "alpha"
        binary_probability = 0.8
        binary_confidence = 0.8
        binary_uncertainty = 0.2
        binary_conformal_set = [1]
        subtype_conformal_set = ["alpha"]
        subtype_top_k = [{"name": "alpha", "probability": 0.8}]
        model_alias = "alias"
        run_id = "run"
        checkpoint_sha256 = "a" * 64
        quality = {"status": "ok", "warnings": []}

    html = btxrd_result_to_html(Result())
    assert "alpha" in html
    assert "checkpoint" in html.lower()
    assert "<html" in html.lower()


def test_robustness_report_marks_unrun_seeds_and_external_validation_unavailable() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_robustness_report(root, seeds=(42, 43), external_root=None)

    assert report["seed_results"][0]["seed"] == 42
    assert report["seed_results"][0]["status"] == "complete"
    assert report["seed_results"][1]["status"] == "not_run"
    assert report["aggregate"]["status"] == "pending"
    assert report["balanced_experiment"]["status"] in {"complete", "not_run"}
    assert report["external_validation"]["status"] == "unavailable"
