"""Validated inference loading for a frozen Phase 8 checkpoint manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from bone_cancer_ai.artifacts.registry import CheckpointRecord, resolve_checkpoint
from bone_cancer_ai.models.hosp_bonenet import HOSPBoneNet, create_hosp_bonenet
from bone_cancer_ai.optimization.contracts import load_yaml, sha256_file
from bone_cancer_ai.optimization.phase8 import phase8_fingerprint


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
    if record.test_status != {"status": "not_run", "planned_phase": 9}:
        raise ValueError("Phase 8 selected artifact has an invalid test status")
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
