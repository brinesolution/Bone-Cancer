"""Validated inference loading for a frozen Phase 8 checkpoint manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from bone_cancer_ai.artifacts.registry import CheckpointRecord, resolve_checkpoint
from bone_cancer_ai.models.hosp_bonenet import HOSPBoneNet, create_hosp_bonenet
from bone_cancer_ai.optimization.contracts import load_yaml, sha256_file
from bone_cancer_ai.optimization.phase8 import phase8_fingerprint

_PHASE9_PROTOCOL = "phase9_frozen_test_v1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_artifact_manifest(root: Path, artifacts: object) -> Mapping[str, str]:
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Phase 9 run status has no artifact manifest")
    for relative, digest in artifacts.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("Phase 9 artifact manifest is malformed")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Phase 9 artifact manifest references a missing artifact")
        if sha256_file(path) != digest:
            raise ValueError("Phase 9 artifact manifest is checksum-invalid")
    return artifacts


def _validate_test_status(
    root: Path,
    status: dict[str, Any],
    *,
    checkpoint_alias: str,
    checkpoint_sha256: str,
) -> None:
    if status == {"status": "not_run", "planned_phase": 9}:
        return
    if (
        status.get("status") != "complete"
        or status.get("phase") != 9
        or not isinstance(status.get("predictions_sha256"), str)
        or not isinstance(status.get("evaluation_run"), str)
    ):
        raise ValueError("Phase 8 selected artifact has an invalid test status")
    run_dir = (root / str(status["evaluation_run"])).resolve()
    if not run_dir.is_relative_to(root / "experiments" / "evaluation"):
        raise ValueError("Phase 9 evaluation run is outside the approved experiment root")
    run_status_path = run_dir / "status.json"
    ledger_path = root / "artifacts" / "metrics" / "hosp_phase9_test_ledger.json"
    predictions_path = root / "artifacts" / "metrics" / "hosp_phase9_test_predictions.csv"
    if not run_status_path.is_file() or not ledger_path.is_file() or not predictions_path.is_file():
        raise ValueError("Phase 9 test provenance is incomplete")
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    digest = str(status["predictions_sha256"])
    if not isinstance(run_status, dict) or not isinstance(ledger, dict):
        raise ValueError("Phase 9 run status or test ledger is malformed")
    contract = ledger.get("contract", {})
    run_contract = run_status.get("contract")
    summary = run_status.get("summary")
    if (
        run_status.get("status") != "complete"
        or run_status.get("protocol") != _PHASE9_PROTOCOL
        or not isinstance(run_contract, dict)
        or run_status.get("fingerprint") != _canonical_digest(run_contract)
        or not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or summary.get("checkpoint_alias") != checkpoint_alias
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("test_evaluation_count") != 1
        or summary.get("visual_qa_status") not in {"complete", "complete_with_limitations"}
        or run_contract.get("contract") != contract
    ):
        raise ValueError("Phase 9 run status is incomplete, stale, or contract-invalid")
    artifacts = _validate_artifact_manifest(root, run_status.get("artifacts"))
    predictions_relative = predictions_path.relative_to(root).as_posix()
    ledger_relative = ledger_path.relative_to(root).as_posix()
    qa_path = root / "artifacts" / "explainability" / "phase9" / "visual_qa.json"
    qa_relative = qa_path.relative_to(root).as_posix()
    if not qa_path.is_file():
        raise ValueError("Phase 9 visual QA artifact is missing")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if (
        ledger.get("status") != "complete"
        or ledger.get("evaluation_count") != 1
        or contract.get("checkpoint_alias") != checkpoint_alias
        or contract.get("checkpoint_sha256") != checkpoint_sha256
        or ledger.get("predictions_sha256") != digest
        or sha256_file(predictions_path) != digest
        or artifacts.get(predictions_relative) != digest
        or artifacts.get(ledger_relative) != sha256_file(ledger_path)
        or artifacts.get(qa_relative) != sha256_file(qa_path)
        or not isinstance(qa, dict)
        or qa.get("status") not in {"complete", "complete_with_limitations"}
    ):
        raise ValueError("Phase 9 test provenance is stale or checksum-invalid")


def load_frozen_phase8_checkpoint(
    project_root: Path,
    alias: str = "hosp_bonenet.phase8.selected",
    *,
    device: str | torch.device = "cpu",
) -> tuple[HOSPBoneNet, dict[str, Any], CheckpointRecord]:
    """Load a selected state through its frozen inference manifest."""
    root = Path(project_root).resolve()
    record = resolve_checkpoint(root, alias)
    if record.phase != "phase_8" or record.model_family != "hosp_bonenet":
        raise ValueError("selected alias is not a Phase 8 HOSP-BoneNet artifact")
    _validate_test_status(
        root,
        record.test_status,
        checkpoint_alias=record.alias,
        checkpoint_sha256=record.checkpoint_sha256,
    )
    run_dir = (root / record.source_run).resolve()
    if not run_dir.is_relative_to(root / "experiments" / "optimization" / "hosp_bonenet"):
        raise ValueError("Phase 8 source run is outside the approved experiment root")
    frozen = load_yaml(run_dir / "final_config.yaml")
    if phase8_fingerprint(frozen) != record.configuration_fingerprint:
        raise ValueError("frozen configuration fingerprint does not match the registry")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    checkpoint_path = record.absolute_checkpoint_path(root)
    if selection.get("checkpoint_sha256") != record.checkpoint_sha256:
        raise ValueError("selection checkpoint checksum does not match the registry")
    if sha256_file(checkpoint_path) != record.checkpoint_sha256:
        raise ValueError("selected checkpoint checksum does not match")
    model_config = frozen.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("frozen model configuration is missing")
    schema = load_yaml(root / "workspace" / "annotations" / "label_schema.yaml")
    model = create_hosp_bonenet(model_config, schema)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("selected checkpoint model state is missing")
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    return model, frozen, record


__all__ = ["load_frozen_phase8_checkpoint"]
