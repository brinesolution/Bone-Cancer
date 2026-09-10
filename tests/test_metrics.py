"""Tests for binary evaluation metrics and report artifacts."""

import json

import pandas as pd
import pytest
import torch

from bone_cancer_ai.evaluation import reports as report_module
from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.evaluation.reports import (
    atomic_write_csv,
    atomic_write_json,
    confusion_matrix_frame,
    measure_inference_latency,
)


def test_binary_metrics_report_perfect_predictions_and_confusion_order():
    metrics = binary_classification_metrics(
        torch.tensor([-4.0, -2.0, 2.0, 4.0]),
        torch.tensor([0, 0, 1, 1]),
    )

    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    for name in (
        "AUROC",
        "AUPRC",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "F1",
        "MCC",
    ):
        assert metrics[name] == pytest.approx(1.0)
    assert 0.0 <= metrics["Brier"] <= 1.0
    assert 0.0 <= metrics["ECE"] <= 1.0


def test_binary_metrics_match_hand_computed_imperfect_predictions():
    probabilities = torch.tensor([0.1, 0.6, 0.4, 0.9], dtype=torch.float64)
    metrics = binary_classification_metrics(
        torch.logit(probabilities),
        torch.tensor([0, 0, 1, 1]),
    )

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "F1",
    ):
        assert metrics[name] == pytest.approx(0.5)
    assert metrics["MCC"] == pytest.approx(0.0)
    assert metrics["AUROC"] == pytest.approx(0.75)
    assert metrics["AUPRC"] == pytest.approx(5 / 6)
    assert metrics["Brier"] == pytest.approx(0.185)
    assert metrics["ECE"] == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("targets", "undefined_metric"),
    [
        ([0, 0, 0], "sensitivity"),
        ([1, 1, 1], "specificity"),
    ],
)
def test_one_class_targets_use_none_for_undefined_metrics(targets, undefined_metric):
    metrics = binary_classification_metrics(
        torch.tensor([-1.0, 0.0, 1.0]),
        torch.tensor(targets),
    )

    assert metrics["AUROC"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics[undefined_metric] is None
    assert metrics["recall"] == metrics["sensitivity"]
    assert metrics["confusion_matrix"] == (
        [[1, 2], [0, 0]] if targets[0] == 0 else [[0, 0], [1, 2]]
    )


def test_all_negative_metrics_keep_undefined_values_json_safe():
    metrics = binary_classification_metrics(
        torch.tensor([-3.0, -2.0]),
        torch.tensor([0, 0]),
    )

    assert metrics["AUPRC"] is None
    assert metrics["precision"] is None
    assert metrics["F1"] is None
    assert metrics["MCC"] is None
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    ("logits", "targets", "threshold", "message"),
    [
        (torch.tensor([]), torch.tensor([]), 0.5, "non-empty"),
        (torch.zeros(1, 2), torch.tensor([0, 1]), 0.5, "one-dimensional"),
        (torch.tensor([0.0]), torch.tensor([0, 1]), 0.5, "same shape"),
        (torch.tensor([float("inf")]), torch.tensor([0]), 0.5, "finite"),
        (torch.tensor([0.0]), torch.tensor([2]), 0.5, "binary"),
        (torch.tensor([0.0]), torch.tensor([0]), -0.1, "threshold"),
        (torch.tensor([0.0]), torch.tensor([0]), float("nan"), "threshold"),
    ],
)
def test_binary_metrics_reject_invalid_inputs(logits, targets, threshold, message):
    with pytest.raises(ValueError, match=message):
        binary_classification_metrics(logits, targets, threshold=threshold)


def test_confusion_matrix_frame_labels_actual_rows_and_predicted_columns():
    frame = confusion_matrix_frame({"confusion_matrix": [[7, 2], [3, 11]]})

    assert frame.index.tolist() == ["actual_negative", "actual_positive"]
    assert frame.columns.tolist() == ["predicted_negative", "predicted_positive"]
    assert frame.values.tolist() == [[7, 2], [3, 11]]


def test_atomic_report_writers_create_parents_and_replace_complete_files(tmp_path):
    json_path = tmp_path / "nested" / "metrics.json"
    csv_path = tmp_path / "nested" / "confusion.csv"

    atomic_write_json({"score": 0.25, "undefined": None}, json_path)
    atomic_write_json({"score": 0.75, "undefined": None}, json_path)
    atomic_write_csv(pd.DataFrame({"count": [1, 2]}), csv_path)
    atomic_write_csv(pd.DataFrame({"count": [3]}), csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "score": 0.75,
        "undefined": None,
    }
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), pd.DataFrame({"count": [3]}))
    assert not list(json_path.parent.glob("*.tmp"))


def test_atomic_json_failure_preserves_existing_report(tmp_path):
    destination = tmp_path / "metrics.json"
    atomic_write_json({"status": "complete"}, destination)

    with pytest.raises(ValueError):
        atomic_write_json({"invalid": float("nan")}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "complete"}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_json_replace_failure_cleans_created_temporary(tmp_path, monkeypatch):
    destination = tmp_path / "metrics.json"
    atomic_write_json({"status": "original"}, destination)
    temporary_paths = []

    def fail_replace(source, target):
        assert source.exists()
        assert target == destination
        temporary_paths.append(source)
        raise OSError("replace failed")

    monkeypatch.setattr(report_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json({"status": "replacement"}, destination)

    assert temporary_paths
    assert all(not path.exists() for path in temporary_paths)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "original"}


def test_atomic_csv_serialization_failure_preserves_report_and_cleans_temporary(tmp_path):
    temporary_paths = []

    class UnserializableValue:
        def __str__(self):
            created = list(tmp_path.glob("*.tmp"))
            assert created
            temporary_paths.extend(created)
            raise RuntimeError("csv serialization failed")

    destination = tmp_path / "history.csv"
    original = pd.DataFrame({"value": [1]})
    atomic_write_csv(original, destination)

    with pytest.raises(RuntimeError, match="csv serialization failed"):
        atomic_write_csv(pd.DataFrame({"value": [UnserializableValue()]}), destination)

    pd.testing.assert_frame_equal(pd.read_csv(destination), original)
    assert temporary_paths
    assert all(not path.exists() for path in temporary_paths)
    assert not list(tmp_path.glob("*.tmp"))


def test_cpu_latency_reports_elapsed_time_and_restores_training_state():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.inference_modes = []

        def forward(self, value):
            self.calls += 1
            self.inference_modes.append((self.training, torch.is_inference_mode_enabled()))
            return value + 1

    model = CountingModel()
    model.train()

    report = measure_inference_latency(
        model,
        torch.zeros(1, 2),
        torch.device("cpu"),
        warmup=2,
        repetitions=3,
    )

    assert model.training is True
    assert model.calls == 5
    assert model.inference_modes == [(False, True)] * 5
    assert report["device"] == "cpu"
    assert report["warmup"] == 2
    assert report["repetitions"] == 3
    assert report["latency_ms"] >= 0.0
    assert report["total_elapsed_ms"] >= report["latency_ms"]
    json.dumps(report, allow_nan=False)


def test_latency_restores_model_state_when_forward_fails():
    class FailingModel(torch.nn.Module):
        def forward(self, value):
            raise RuntimeError("forward failed")

    model = FailingModel()
    model.train()

    with pytest.raises(RuntimeError, match="forward failed"):
        measure_inference_latency(model, torch.zeros(1), "cpu", warmup=1, repetitions=1)

    assert model.training is True


@pytest.mark.parametrize("fail", [False, True])
def test_latency_restores_exact_mixed_module_states(fail):
    class MixedModeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_norm = torch.nn.BatchNorm1d(2)
            self.fail = fail

        def forward(self, value):
            if self.fail:
                raise RuntimeError("forward failed")
            return self.batch_norm(value)

    model = MixedModeModel()
    model.train()
    model.batch_norm.eval()

    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            measure_inference_latency(model, torch.zeros(2, 2), "cpu", warmup=1, repetitions=1)
    else:
        measure_inference_latency(model, torch.zeros(2, 2), "cpu", warmup=1, repetitions=1)

    assert model.training is True
    assert model.batch_norm.training is False


@pytest.mark.parametrize(
    ("warmup", "repetitions"),
    [(0, 1), (1, 0), (True, 1), (1, 1.5)],
)
def test_latency_requires_positive_integer_run_counts(warmup, repetitions):
    with pytest.raises(ValueError, match="positive integers"):
        measure_inference_latency(
            torch.nn.Identity(),
            torch.zeros(1),
            "cpu",
            warmup=warmup,
            repetitions=repetitions,
        )
