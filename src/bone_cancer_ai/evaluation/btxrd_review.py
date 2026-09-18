"""Post-training integrity review for the completed BTXRD Phase B run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bone_cancer_ai.artifacts.registry import resolve_checkpoint
from bone_cancer_ai.optimization.contracts import sha256_file

DEFAULT_ALIAS = "hosp_bonenet.btxrd.subtype.selected"


def review_btxrd_run(
    project_root: Path,
    *,
    alias: str = DEFAULT_ALIAS,
) -> dict[str, Any]:
    """Verify the registry, checkpoint, protocol, and reported sample counts."""

    root = Path(project_root).resolve()
    record = resolve_checkpoint(root, alias)
    if record.phase != "phase_b" or record.model_family != "hosp_bonenet":
        raise ValueError("registry record is not the BTXRD HOSP Phase B artifact")
    if record.test_status.get("status") != "complete":
        raise ValueError("BTXRD test status is not complete")
    checkpoint = record.absolute_checkpoint_path(root)
    if sha256_file(checkpoint) != record.checkpoint_sha256:
        raise ValueError("BTXRD checkpoint checksum mismatch")
    run_dir = Path(record.source_run)
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if (
        not isinstance(metrics, dict)
        or not isinstance(status, dict)
        or status.get("status") != "complete"
    ):
        raise ValueError("BTXRD run status is not complete")
    if metrics.get("protocol") != "btxrd_fresh_calibration_and_test":
        raise ValueError("BTXRD metrics protocol is not fresh calibration and test")
    if metrics.get("configuration_fingerprint") != record.configuration_fingerprint:
        raise ValueError("BTXRD run fingerprint does not match the registry")
    evaluation = metrics.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("BTXRD evaluation section is missing")
    binary = evaluation.get("binary")
    subtype = evaluation.get("subtype")
    if not isinstance(binary, dict) or not isinstance(subtype, dict):
        raise ValueError("BTXRD binary or subtype metrics are missing")
    binary_samples = int(sum(sum(row) for row in binary["confusion_matrix"]))
    subtype_samples = int(subtype["sample_count"])
    subtype_classes = len(subtype["classes_present"])
    if binary_samples < 1 or subtype_samples < 1 or subtype_classes < 2:
        raise ValueError("BTXRD metrics contain invalid sample counts")
    return {
        "status": "pass",
        "alias": record.alias,
        "phase": record.phase,
        "model_family": record.model_family,
        "run_id": record.source_run,
        "checkpoint_sha256": record.checkpoint_sha256,
        "configuration_fingerprint": record.configuration_fingerprint,
        "fresh_evaluation": True,
        "binary_test_samples": binary_samples,
        "binary_auroc": float(binary["AUROC"]),
        "subtype_test_samples": subtype_samples,
        "subtype_classes": subtype_classes,
        "subtype_accuracy": float(subtype["accuracy"]),
        "subtype_macro_f1": float(subtype["macro_f1"]),
        "binary_conformal_coverage": float(evaluation["binary_conformal"]["coverage"]),
        "subtype_conformal_coverage": float(evaluation["subtype_conformal"]["coverage"]),
    }


__all__ = ["DEFAULT_ALIAS", "review_btxrd_run"]
