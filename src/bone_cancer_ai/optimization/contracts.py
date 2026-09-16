"""Configuration and artifact contracts for Phase 8 optimization."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from bone_cancer_ai.optimization.phase8 import ensure_validation_only

CONFIG_KEYS = {
    "phase", "output_root", "parent_alias", "model_config", "data_config",
    "schema_path", "split_dir", "allowed_splits", "selection_split",
    "selection_metric", "device", "amp", "seed", "data_screen",
    "optimization", "ablations", "ablation_run", "final_run",
}
SUPPORTED_ABLATIONS = {
    "B0": (False, "none"),
    "B1": (True, "attention"),
    "B2": (True, "gated"),
}
METRIC_COLUMNS = (
    "AUROC", "AUPRC", "balanced_accuracy", "sensitivity", "F1", "ECE", "Brier"
)


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value: object, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict JSON data") from error


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(strict_json(value, name=str(path)), indent=2, sort_keys=True)
    atomic_text(path, f"{payload}\n")


def write_yaml(path: Path, value: object) -> None:
    payload = yaml.safe_dump(strict_json(value, name=str(path)), sort_keys=False)
    atomic_text(path, payload)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return loaded


def _positive_int(value: object, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_phase8_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the tracked Phase 8 configuration."""
    document = load_yaml(Path(path))
    if set(document) != CONFIG_KEYS:
        raise ValueError(
            "Phase 8 config keys are invalid; "
            f"missing={sorted(CONFIG_KEYS - set(document))}, "
            f"extra={sorted(set(document) - CONFIG_KEYS)}"
        )
    ensure_validation_only(document)
    if document["phase"] != 8:
        raise ValueError("phase must be 8")
    if document["allowed_splits"] != ["fit_train", "val"]:
        raise ValueError("allowed_splits must be exactly fit_train and val")
    if document["selection_split"] != "val" or document["selection_metric"] != "AUROC":
        raise ValueError("Phase 8 selection must use validation AUROC")
    if document["device"] != "cuda" or document["amp"] is not True:
        raise ValueError("Phase 8 requires CUDA AMP")
    if type(document["seed"]) is not int or document["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if document["optimization"].get("trials") != 10:
        raise ValueError("Phase 8 compact search must contain exactly 10 trials")
    ablations = document["ablations"]
    if not isinstance(ablations, list) or [row.get("id") for row in ablations] != [
        "B0", "B1", "B2"
    ]:
        raise ValueError("only supported B0, B1, and B2 ablations may be configured")
    for section_name in ("data_screen", "optimization", "ablation_run", "final_run"):
        section = document[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")
        _positive_int(section.get("epochs"), f"{section_name}.epochs")
        _positive_int(
            section.get("early_stopping_patience"),
            f"{section_name}.early_stopping_patience",
        )
        _positive_int(
            section.get("train_samples"), f"{section_name}.train_samples", allow_none=True
        )
        _positive_int(
            section.get("val_samples"), f"{section_name}.val_samples", allow_none=True
        )
    return strict_json(document, name="Phase 8 config")


def ablation_model_config(
    base: Mapping[str, Any], ablation: str, *, dropout: float
) -> dict[str, Any]:
    """Resolve one supported binary-only architecture switch set."""
    try:
        localization, spatial_path = SUPPORTED_ABLATIONS[ablation]
    except KeyError as error:
        raise ValueError(f"unsupported ablation: {ablation}") from error
    resolved = dict(base)
    optional = ("subtype_enabled", "risk_enabled", "concepts_enabled")
    if any(bool(resolved.get(key)) for key in optional):
        raise ValueError("optional supervision must remain disabled in Phase 8")
    resolved.update(
        {
            "pretrained": False,
            "dropout": float(dropout),
            "localization_enabled": localization,
            "spatial_path": spatial_path,
            "subtype_enabled": False,
            "num_subtypes": 0,
            "risk_enabled": False,
            "num_risk_classes": 0,
            "concepts_enabled": False,
            "num_concepts": 0,
        }
    )
    return resolved


def load_completed_stage(stage_dir: Path, signature: str) -> dict[str, Any] | None:
    """Return a complete stage only after signature and checkpoint verification."""
    metrics_path = Path(stage_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"stage metrics are malformed: {metrics_path}") from error
    if not isinstance(metrics, dict) or metrics.get("status") != "complete":
        raise ValueError(f"stage metrics do not identify a complete stage: {metrics_path}")
    if metrics.get("signature") != signature:
        raise ValueError("completed stage signature does not match")
    checkpoint_name = metrics.get("best_checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("completed stage checkpoint path is invalid")
    checkpoint = Path(stage_dir) / checkpoint_name
    if not checkpoint.is_file():
        raise ValueError("completed stage checkpoint is missing")
    if sha256_file(checkpoint) != metrics.get("checkpoint_sha256"):
        raise ValueError("completed stage checkpoint checksum does not match")
    if not isinstance(metrics.get("validation"), dict):
        raise ValueError("completed stage validation metrics are missing")
    return metrics


__all__ = [
    "METRIC_COLUMNS",
    "SUPPORTED_ABLATIONS",
    "ablation_model_config",
    "load_completed_stage",
    "load_yaml",
    "resolve_phase8_config",
    "sha256_file",
    "strict_json",
    "write_json",
    "write_yaml",
]
