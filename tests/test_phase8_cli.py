"""Phase 8 orchestration contracts that do not launch GPU training."""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bone_cancer_ai.optimization import workflow


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
    metrics = {
        "status": "complete",
        "signature": "expected",
        "best_checkpoint": "best.pt",
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "validation": {"AUROC": 0.9},
    }
    (stage / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    assert module.load_completed_stage(stage, "expected") == metrics
    with pytest.raises(ValueError, match="signature"):
        module.load_completed_stage(stage, "different")

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        module.load_completed_stage(stage, "expected")


def test_phase8_context_rejects_noncanonical_config_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="canonical.*optimization.yaml"):
        workflow._context(tmp_path / "other.yaml", {})


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
