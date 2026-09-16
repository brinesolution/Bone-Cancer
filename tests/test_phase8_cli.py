"""Phase 8 orchestration contracts that do not launch GPU training."""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.optimization import execution, workflow


def _module():
    path = Path(__file__).parents[1] / "scripts" / "11_tune_hosp_bonenet.py"
    spec = importlib.util.spec_from_file_location("phase8_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_phase8_config_is_validation_only_and_compact() -> None:
    module = _module()
    config = module.resolve_phase8_config(
        Path(__file__).parents[1] / "configs" / "optimization.yaml"
    )

    assert config["allowed_splits"] == ["fit_train", "val"]
    assert config["optimization"]["trials"] == 10
    assert config["selection_metric"] == "AUROC"
    assert [row["id"] for row in config["ablations"]] == ["B0", "B1", "B2"]
    assert all(row["id"] not in {"B3", "B4", "B5"} for row in config["ablations"])


@pytest.mark.parametrize(
    ("ablation", "localization", "spatial_path"),
    [("B0", False, "none"), ("B1", True, "attention"), ("B2", True, "gated")],
)
def test_ablation_model_config_maps_only_supported_binary_paths(
    ablation: str, localization: bool, spatial_path: str
) -> None:
    module = _module()
    base = {
        "name": "hosp_bonenet",
        "localization_enabled": True,
        "subtype_enabled": False,
        "risk_enabled": False,
        "concepts_enabled": False,
    }

    resolved = module.ablation_model_config(base, ablation, dropout=0.2)

    assert resolved["localization_enabled"] is localization
    assert resolved["spatial_path"] == spatial_path
    assert resolved["subtype_enabled"] is False
    assert resolved["risk_enabled"] is False
    assert resolved["concepts_enabled"] is False


def test_completed_stage_requires_matching_signature_and_checkpoint_digest(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / "stage"
    stage.mkdir()
    checkpoint = stage / "best.pt"
    torch.save({"model_state": {}}, checkpoint)
    (stage / "config.yaml").write_text("stage: synthetic\n", encoding="utf-8")
    pd.DataFrame([{"epoch": 1, "val_AUROC": 1.0}]).to_csv(
        stage / "history.csv", index=False
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["negative", "positive"],
            "target": [0, 1],
            "logit": [-2.0, 2.0],
            "probability": torch.sigmoid(torch.tensor([-2.0, 2.0])).tolist(),
        }
    )
    predictions.to_csv(stage / "validation_predictions.csv", index=False)
    validation = binary_classification_metrics(predictions["logit"], predictions["target"])
    metrics = {
        "status": "complete",
        "signature": "expected",
        "best_checkpoint": "best.pt",
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "config_sha256": module.sha256_file(stage / "config.yaml"),
        "history_sha256": module.sha256_file(stage / "history.csv"),
        "predictions_sha256": module.sha256_file(stage / "validation_predictions.csv"),
        "validation": validation,
    }
    (stage / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    assert module.load_completed_stage(stage, "expected") == metrics
    with pytest.raises(ValueError, match="signature"):
        module.load_completed_stage(stage, "different")

    metrics["validation"]["Brier"] += 5e-9
    (stage / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert module.load_completed_stage(stage, "expected") == metrics

    metrics["validation"]["Brier"] += 1e-5
    (stage / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="Brier"):
        module.load_completed_stage(stage, "expected")

    metrics["validation"] = validation
    (stage / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        module.load_completed_stage(stage, "expected")


def test_phase8_context_rejects_noncanonical_config_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="canonical.*optimization.yaml"):
        workflow._context(tmp_path / "other.yaml", {})


def test_phase8_source_fingerprint_covers_transitive_model_and_preprocessing_code() -> None:
    root = Path(__file__).parents[1]

    paths = set(workflow._phase8_source_paths(root).values())

    assert root / "src/bone_cancer_ai/models/hosp_bonenet/encoder.py" in paths
    assert root / "src/bone_cancer_ai/models/hosp_bonenet/fusion.py" in paths
    assert root / "src/bone_cancer_ai/data/preprocessing.py" in paths
    assert root / "src/bone_cancer_ai/losses/total.py" in paths


def test_validated_training_stage_binds_expected_definition(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    definition = {"architecture": "B2", "data": {"image_size": 384}}
    workflow.write_yaml(stage / "config.yaml", definition)
    checkpoint = stage / "best.pt"
    torch.save({"model_state": {}}, checkpoint)
    pd.DataFrame([{"epoch": 1, "val_AUROC": 1.0}]).to_csv(
        stage / "history.csv", index=False
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["negative", "positive"],
            "target": [0, 1],
            "logit": [-2.0, 2.0],
            "probability": torch.sigmoid(torch.tensor([-2.0, 2.0])).tolist(),
        }
    )
    predictions.to_csv(stage / "validation_predictions.csv", index=False)
    metrics = {
        "status": "complete",
        "signature": workflow.phase8_fingerprint(definition),
        "architecture": "B2",
        "best_checkpoint": "best.pt",
        "checkpoint_sha256": workflow.sha256_file(checkpoint),
        "config_sha256": workflow.sha256_file(stage / "config.yaml"),
        "history_sha256": workflow.sha256_file(stage / "history.csv"),
        "predictions_sha256": workflow.sha256_file(stage / "validation_predictions.csv"),
        "validation": binary_classification_metrics(
            predictions["logit"], predictions["target"]
        ),
    }
    (stage / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    assert workflow._validated_training_stage(stage, definition) == metrics
    with pytest.raises(ValueError, match="definition differs"):
        workflow._validated_training_stage(
            stage, {"architecture": "B1", "data": {"image_size": 384}}
        )


def test_pruned_trial_cleanup_removes_only_terminal_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "best.pt").write_bytes(b"best")
    (tmp_path / "last.pt").write_bytes(b"last")
    (tmp_path / "status.json").write_text("{}", encoding="utf-8")

    workflow._remove_pruned_checkpoints(tmp_path)

    assert not (tmp_path / "best.pt").exists()
    assert not (tmp_path / "last.pt").exists()
    assert (tmp_path / "status.json").exists()


def test_incumbent_reuse_requires_status_prediction_hash_and_recomputed_metrics(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "incumbent"
    stage.mkdir()
    settings = execution.base_settings()
    parent_sha = "a" * 64
    context = {
        "base_model": {"name": "hosp_bonenet"},
        "input_hashes": {"source": "b" * 64},
        "parent_record": SimpleNamespace(checkpoint_sha256=parent_sha),
    }
    signature = workflow.phase8_fingerprint(
        {
            "kind": "incumbent_validation_recompute",
            "model": context["base_model"],
            "data": settings,
            "parent_sha256": parent_sha,
            "inputs": context["input_hashes"],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["negative", "positive"],
            "target": [0, 1],
            "logit": [-2.0, 2.0],
            "probability": torch.sigmoid(torch.tensor([-2.0, 2.0])).tolist(),
        }
    )
    predictions.to_csv(stage / "validation_predictions.csv", index=False)
    validation = binary_classification_metrics(predictions["logit"], predictions["target"])
    payload = {
        "status": "complete",
        "signature": signature,
        "architecture": "B2",
        "validation": validation,
        "checkpoint_sha256": parent_sha,
        "predictions_sha256": workflow.sha256_file(stage / "validation_predictions.csv"),
    }
    (stage / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (stage / "status.json").write_text(
        json.dumps({"status": "complete", "signature": signature}), encoding="utf-8"
    )

    assert execution.evaluation_stage(context, stage, settings) == payload

    (stage / "status.json").write_text(
        json.dumps({"status": "running", "signature": signature}), encoding="utf-8"
    )
    assert execution.evaluation_stage(context, stage, settings) == payload
    assert json.loads((stage / "status.json").read_text(encoding="utf-8")) == {
        "status": "complete",
        "signature": signature,
    }

    (stage / "validation_predictions.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction.*checksum"):
        execution.evaluation_stage(context, stage, settings)


def test_ablation_table_is_bound_to_stage_and_selection_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = {
        name: {
            "checkpoint_sha256": name.lower() * 32,
            "validation": {
                metric: float(index + 1) / 10
                for index, metric in enumerate(workflow.METRIC_COLUMNS)
            },
        }
        for name in ("B0", "B1", "B2")
    }
    monkeypatch.setattr(
        workflow,
        "training_stage_definition",
        lambda context, architecture, **kwargs: {"architecture": architecture},
    )
    monkeypatch.setattr(
        workflow,
        "_validated_training_stage",
        lambda stage, expected: metrics[stage.name],
    )
    selection = {
        "candidate": "phase8_optimized",
        "checkpoint_sha256": "f" * 64,
        "validation": {
            metric: float(index + 2) / 10
            for index, metric in enumerate(workflow.METRIC_COLUMNS)
        },
    }
    rows = []
    components = {"B0": "plain", "B1": "localization", "B2": "fusion"}
    for name, component in components.items():
        row = {
            "ablation": name,
            "component": component,
            "protocol": "phase7_warm_start_branch_ablation",
            "initialization": "hosp_bonenet.phase7.selected",
            "checkpoint_sha256": metrics[name]["checkpoint_sha256"],
            "test_status": "not_run",
            **metrics[name]["validation"],
        }
        rows.append(row)
    rows.append(
        {
            "ablation": "FULL",
            "component": "selected supported binary configuration",
            "protocol": selection["candidate"],
            "initialization": float("nan"),
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "test_status": "not_run",
            **selection["validation"],
        }
    )
    frame = pd.DataFrame(rows)
    context = {
        "config": {
            "ablations": [
                {"id": name, "component": component}
                for name, component in components.items()
            ],
            "ablation_run": {
                "epochs": 1,
                "early_stopping_patience": 1,
                "train_samples": 2,
                "val_samples": 2,
            },
        },
        "parent_record": SimpleNamespace(alias="hosp_bonenet.phase7.selected"),
    }

    workflow._validate_ablation_table(
        context,
        tmp_path,
        frame,
        selection,
        settings=execution.base_settings(),
        hparams=execution.base_hparams(),
    )

    frame.loc[frame["ablation"] == "B1", "Brier"] += 0.01
    with pytest.raises(ValueError, match="B1.*Brier"):
        workflow._validate_ablation_table(
            context,
            tmp_path,
            frame,
            selection,
            settings=execution.base_settings(),
            hparams=execution.base_hparams(),
        )


def test_publication_failure_restores_alias_and_global_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "experiments/optimization/hosp_bonenet/run"
    run.mkdir(parents=True)
    (run / "selected.pt").write_bytes(b"checkpoint")
    (run / "final_config.yaml").write_text("new-config\n", encoding="utf-8")
    (run / "hosp_ablation.csv").write_text("new-metrics\n", encoding="utf-8")
    global_config = tmp_path / "configs/hosp_bonenet_phase8_final.yaml"
    global_metrics = tmp_path / "artifacts/metrics/hosp_ablation.csv"
    global_config.parent.mkdir(parents=True)
    global_metrics.parent.mkdir(parents=True)
    global_config.write_text("old-config\n", encoding="utf-8")
    global_metrics.write_text("old-metrics\n", encoding="utf-8")
    selection = {
        "validation": {"AUROC": 0.9},
        "configuration_fingerprint": "a" * 64,
    }
    promoted = SimpleNamespace(alias="hosp_bonenet.phase8.selected")
    promotion = SimpleNamespace(promoted=promoted, previous=None)
    restored: list[tuple[object, object]] = []
    real_atomic_text = workflow.atomic_text

    monkeypatch.setattr(workflow, "_validate_local_bundle", lambda *args, **kwargs: selection)
    monkeypatch.setattr(workflow, "checkpoint_publication_lock", lambda *args: nullcontext())
    monkeypatch.setattr(
        workflow, "promote_checkpoint_transaction", lambda *args, **kwargs: promotion
    )
    monkeypatch.setattr(workflow, "resolve_checkpoint", lambda *args: promoted)
    monkeypatch.setattr(
        workflow,
        "restore_checkpoint_alias",
        lambda *args, **kwargs: restored.append((kwargs["expected_current"], kwargs["previous"])),
    )

    calls = 0

    def fail_second_global_write(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        real_atomic_text(path, text)

    monkeypatch.setattr(workflow, "atomic_text", fail_second_global_write)

    with pytest.raises(OSError, match="publication failed"):
        workflow._publish_selection({"root": tmp_path}, run, "fingerprint")

    assert restored == [(promoted, None)]
    assert global_config.read_text(encoding="utf-8") == "old-config\n"
    assert global_metrics.read_text(encoding="utf-8") == "old-metrics\n"
