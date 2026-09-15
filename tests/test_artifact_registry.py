"""Behavioral contracts for verified checkpoint promotion and reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bone_cancer_ai.artifacts.registry import promote_checkpoint, resolve_checkpoint


def _promotion_details() -> dict[str, object]:
    return {
        "model_family": "hosp_bonenet",
        "phase": "phase_7",
        "source_run": "experiments/hosp_bonenet/run-001",
        "selected_metric": {"split": "val", "name": "AUROC", "value": 0.91},
        "configuration_fingerprint": "a" * 64,
        "test_status": {"status": "not_run", "planned_phase": 9},
    }


def _checkpoint(project_root: Path, payload: bytes = b"verified checkpoint") -> Path:
    checkpoint = project_root / "experiments" / "hosp_bonenet" / "run-001" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    return checkpoint


def test_promotion_stores_a_project_relative_sha256_verified_checkpoint(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)

    record = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    promoted = project_root / record.checkpoint_path
    assert record.alias == "hosp-selected"
    assert record.source_run == "experiments/hosp_bonenet/run-001"
    assert record.checkpoint_sha256 == hashlib.sha256(b"verified checkpoint").hexdigest()
    assert promoted.read_bytes() == b"verified checkpoint"
    assert promoted.is_relative_to(project_root)
    registry = json.loads(
        (project_root / "artifacts" / "checkpoints" / "registry.json").read_text(encoding="utf-8")
    )
    assert registry["hosp-selected"]["checkpoint_path"] == record.checkpoint_path


def test_alias_resolution_returns_the_verified_promoted_checkpoint(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    promoted = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    resolved = resolve_checkpoint(project_root, "hosp-selected")

    assert resolved == promoted
    assert resolved.absolute_checkpoint_path(project_root) == (
        project_root / promoted.checkpoint_path
    )


def test_repeating_the_same_promotion_is_idempotent(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    first = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    registry_path = project_root / "artifacts" / "checkpoints" / "registry.json"
    registry_before = registry_path.read_bytes()

    second = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    assert second == first
    assert registry_path.read_bytes() == registry_before


def test_alias_resolution_rejects_a_tampered_promoted_checkpoint(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    record = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    (project_root / record.checkpoint_path).write_bytes(b"tampered checkpoint")

    with pytest.raises(ValueError, match="checksum"):
        resolve_checkpoint(project_root, "hosp-selected")


def test_alias_resolution_rejects_a_registry_path_outside_the_checkpoint_store(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    registry_path = project_root / "artifacts" / "checkpoints" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["hosp-selected"]["checkpoint_path"] = "experiments/hosp_bonenet/run-001/best.pt"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint path"):
        resolve_checkpoint(project_root, "hosp-selected")


def test_promotion_rejects_a_checkpoint_from_the_immutable_dataset(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "dataset" / "Main" / "best.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source data must not be linked")
    details = _promotion_details()
    details["source_run"] = "dataset/Main"

    with pytest.raises(ValueError, match="dataset/Main"):
        promote_checkpoint(
            project_root,
            alias="hosp-selected",
            checkpoint_path=source,
            **details,
        )
