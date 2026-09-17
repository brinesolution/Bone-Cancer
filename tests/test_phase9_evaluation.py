"""Phase 9 statistical evaluation and comparison contracts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from bone_cancer_ai.evaluation.bootstrap import bootstrap_confidence_intervals
from bone_cancer_ai.evaluation.comparison import (
    ensure_matching_cohort,
    plot_final_comparison,
    validate_prediction_frame,
)
from bone_cancer_ai.evaluation.phase9 import (
    _validate_completed_run,
    _validate_visual_qa,
    seal_or_reuse_test_predictions,
)
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator
from bone_cancer_ai.optimization.frozen import _validate_test_status


def _predictions() -> pd.DataFrame:
    probabilities = np.asarray([0.05, 0.20, 0.40, 0.70, 0.85, 0.95])
    return pd.DataFrame(
        {
            "sample_id": [f"sample_{index}" for index in range(6)],
            "target": [0, 0, 0, 1, 1, 1],
            "logit": np.log(probabilities / (1.0 - probabilities)),
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
        }
    )


def test_bootstrap_intervals_are_reproducible_and_contain_estimates() -> None:
    frame = _predictions()
    first = bootstrap_confidence_intervals(
        frame["logit"], frame["target"], resamples=100, confidence=0.90, seed=17
    )
    second = bootstrap_confidence_intervals(
        frame["logit"], frame["target"], resamples=100, confidence=0.90, seed=17
    )

    assert first == second
    assert set(first) == {"AUROC", "AUPRC", "balanced_accuracy", "F1"}
    for interval in first.values():
        assert interval["lower"] <= interval["estimate"] <= interval["upper"]
        assert interval["valid_resamples"] == 100
        json.dumps(interval, allow_nan=False)


@pytest.mark.parametrize(
    ("resamples", "confidence"), [(0, 0.95), (True, 0.95), (10, 0.0), (10, 1.0)]
)
def test_bootstrap_rejects_invalid_settings(resamples, confidence) -> None:
    frame = _predictions()
    with pytest.raises(ValueError):
        bootstrap_confidence_intervals(
            frame["logit"], frame["target"], resamples=resamples, confidence=confidence
        )


def test_binary_conformal_sets_are_deterministic_and_report_coverage() -> None:
    calibrator = BinaryConformalCalibrator.fit(
        probabilities=np.asarray([0.4, 0.5, 0.5, 0.6]),
        targets=np.asarray([0, 0, 1, 1]),
        alpha=0.25,
    )
    sets = calibrator.predict_sets(np.asarray([0.05, 0.5, 0.95]))
    summary = calibrator.summarize(
        np.asarray([0.05, 0.5, 0.95]), np.asarray([0, 1, 1])
    )

    assert sets == [[0], [0, 1], [1]]
    assert calibrator.calibration_count == 4
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["mean_set_size"] == pytest.approx(4 / 3)
    assert summary["ambiguous_count"] == 1


def test_prediction_validation_and_matching_cohort_detect_tampering(tmp_path) -> None:
    frame = _predictions()
    checked = validate_prediction_frame(frame, label="model")
    assert len(checked) == 6
    ensure_matching_cohort({"a": checked, "b": checked.copy()})

    changed = checked.copy()
    changed.loc[0, "target"] = 1
    with pytest.raises(ValueError, match="targets"):
        ensure_matching_cohort({"a": checked, "b": changed})

    changed = checked.copy()
    changed.loc[0, "probability"] = 0.8
    with pytest.raises(ValueError, match="probability"):
        validate_prediction_frame(changed, label="changed")

    outputs = plot_final_comparison({"model": checked}, tmp_path)
    assert outputs["roc"].is_file()
    assert outputs["pr"].is_file()
    assert outputs["calibration"].is_file()
    assert outputs["confusion_matrices"][0].is_file()


def test_sealed_test_predictions_are_reused_without_calling_predictor(tmp_path) -> None:
    expected = _predictions()[["sample_id", "target"]]
    calls = []

    def predict() -> pd.DataFrame:
        calls.append("called")
        return _predictions()

    first = seal_or_reuse_test_predictions(
        tmp_path,
        contract={"checkpoint_sha256": "a" * 64, "test_manifest_sha256": "b" * 64},
        expected=expected,
        predictor=predict,
    )
    second = seal_or_reuse_test_predictions(
        tmp_path,
        contract={"checkpoint_sha256": "a" * 64, "test_manifest_sha256": "b" * 64},
        expected=expected,
        predictor=lambda: pytest.fail("sealed test inference must not run twice"),
    )

    assert calls == ["called"]
    assert first.reused is False
    assert second.reused is True
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert second.ledger["evaluation_count"] == 1


def test_sealed_test_predictions_reject_checksum_change(tmp_path) -> None:
    expected = _predictions()[["sample_id", "target"]]
    seal_or_reuse_test_predictions(
        tmp_path,
        contract={"checkpoint_sha256": "a" * 64},
        expected=expected,
        predictor=_predictions,
    )
    path = tmp_path / "hosp_phase9_test_predictions.csv"
    path.write_text(path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        seal_or_reuse_test_predictions(
            tmp_path,
            contract={"checkpoint_sha256": "a" * 64},
            expected=expected,
            predictor=lambda: pytest.fail("tampered sealed output must fail closed"),
        )


def test_incomplete_test_attempt_blocks_automatic_repeat_inference(tmp_path) -> None:
    expected = _predictions()[["sample_id", "target"]]

    def fail_after_attempt() -> pd.DataFrame:
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        seal_or_reuse_test_predictions(
            tmp_path,
            contract={"checkpoint_sha256": "a" * 64},
            expected=expected,
            predictor=fail_after_attempt,
        )
    with pytest.raises(RuntimeError, match="automatic repeat inference is blocked"):
        seal_or_reuse_test_predictions(
            tmp_path,
            contract={"checkpoint_sha256": "a" * 64},
            expected=expected,
            predictor=lambda: pytest.fail("an incomplete attempt must fail closed"),
        )


def test_completed_run_reuse_requires_exact_contract(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    qa = tmp_path / "artifacts" / "explainability" / "phase9" / "visual_qa.json"
    qa.parent.mkdir(parents=True)
    qa.write_text(
        json.dumps(
            {
                "status": "complete_with_limitations",
                "review_type": "technical visual QA",
                "checks": {"alignment": "PASS"},
                "conclusion": "Qualitative debugging only.",
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract = {"conformal": {"alpha": 0.1}, "gallery": {"per_category": 4}}
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "phase9_frozen_test_v1",
                "fingerprint": fingerprint,
                "contract": contract,
                "summary": {},
                "artifacts": {
                    "artifact.json": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "artifacts/explainability/phase9/visual_qa.json": hashlib.sha256(
                        qa.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    _validate_completed_run(
        tmp_path,
        run_dir,
        expected_contract=contract,
        expected_fingerprint=fingerprint,
    )
    with pytest.raises(ValueError, match="requested contract"):
        _validate_completed_run(
            tmp_path,
            run_dir,
            expected_contract={**contract, "gallery": {"per_category": 8}},
            expected_fingerprint=fingerprint,
        )


def test_visual_qa_must_be_explicitly_completed(tmp_path) -> None:
    qa = tmp_path / "visual_qa.json"
    qa.write_text('{"status":"pending_visual_inspection"}', encoding="utf-8")
    with pytest.raises(ValueError, match="has not been completed"):
        _validate_visual_qa(qa)


def test_frozen_loader_test_status_requires_complete_checksum_provenance(tmp_path) -> None:
    run_dir = tmp_path / "experiments" / "evaluation" / "phase9-test"
    metrics = tmp_path / "artifacts" / "metrics"
    qa_path = tmp_path / "artifacts" / "explainability" / "phase9" / "visual_qa.json"
    run_dir.mkdir(parents=True)
    metrics.mkdir(parents=True)
    qa_path.parent.mkdir(parents=True)
    predictions = metrics / "hosp_phase9_test_predictions.csv"
    predictions.write_text("sample_id,target\ns1,0\n", encoding="utf-8")
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    checkpoint_alias = "hosp_bonenet.phase8.selected"
    checkpoint_sha256 = "a" * 64
    ledger_path = metrics / "hosp_phase9_test_ledger.json"
    ledger_contract = {
        "checkpoint_alias": checkpoint_alias,
        "checkpoint_sha256": checkpoint_sha256,
    }
    ledger_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_count": 1,
                "predictions_sha256": digest,
                "contract": ledger_contract,
            }
        ),
        encoding="utf-8",
    )
    qa_path.write_text(
        json.dumps({"status": "complete_with_limitations"}), encoding="utf-8"
    )
    status = {
        "status": "complete",
        "phase": 9,
        "evaluation_run": "experiments/evaluation/phase9-test",
        "predictions_sha256": digest,
    }

    (run_dir / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="run status"):
        _validate_test_status(
            tmp_path,
            status,
            checkpoint_alias=checkpoint_alias,
            checkpoint_sha256=checkpoint_sha256,
        )

    run_contract = {"contract": ledger_contract, "evaluation_config": {"phase": 9}}
    fingerprint = hashlib.sha256(
        json.dumps(run_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = {
        predictions.relative_to(tmp_path).as_posix(): digest,
        ledger_path.relative_to(tmp_path).as_posix(): hashlib.sha256(
            ledger_path.read_bytes()
        ).hexdigest(),
        qa_path.relative_to(tmp_path).as_posix(): hashlib.sha256(qa_path.read_bytes()).hexdigest(),
    }
    run_status = {
        "status": "complete",
        "protocol": "phase9_frozen_test_v1",
        "fingerprint": fingerprint,
        "contract": run_contract,
        "summary": {
            "status": "complete",
            "checkpoint_alias": checkpoint_alias,
            "checkpoint_sha256": checkpoint_sha256,
            "test_evaluation_count": 1,
            "visual_qa_status": "complete_with_limitations",
        },
        "artifacts": artifacts,
    }
    (run_dir / "status.json").write_text(json.dumps(run_status), encoding="utf-8")
    _validate_test_status(
        tmp_path,
        status,
        checkpoint_alias=checkpoint_alias,
        checkpoint_sha256=checkpoint_sha256,
    )

    changed_status = {**run_status, "summary": {**run_status["summary"], "checkpoint_alias": "x"}}
    (run_dir / "status.json").write_text(json.dumps(changed_status), encoding="utf-8")
    with pytest.raises(ValueError, match="contract-invalid"):
        _validate_test_status(
            tmp_path,
            status,
            checkpoint_alias=checkpoint_alias,
            checkpoint_sha256=checkpoint_sha256,
        )

    (run_dir / "status.json").write_text(json.dumps(run_status), encoding="utf-8")
    predictions.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _validate_test_status(
            tmp_path,
            status,
            checkpoint_alias=checkpoint_alias,
            checkpoint_sha256=checkpoint_sha256,
        )
