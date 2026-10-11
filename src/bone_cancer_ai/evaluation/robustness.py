"""Honest bookkeeping for multi-seed and external BTXRD validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _completed_seed(root: Path, seed: int) -> dict[str, Any] | None:
    """Find the newest complete, non-smoke run for one training seed."""

    candidates = sorted(
        (root / "experiments" / "btxrd_subtype").glob("run-*/contract.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for contract_path in candidates:
        run = contract_path.parent
        metrics_path = run / "metrics.json"
        status_path = run / "status.json"
        if not metrics_path.is_file() or not status_path.is_file():
            continue
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        runtime = contract.get("runtime", {})
        actual_seed = runtime.get("training", {}).get("seed")
        execution = contract.get("execution", {})
        try:
            parsed_seed = int(actual_seed)
        except (TypeError, ValueError):
            continue
        if (
            status.get("status") != "complete"
            or parsed_seed != seed
            or execution.get("max_batches") is not None
            or execution.get("epochs_override") is not None
        ):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        evaluation = metrics.get("evaluation", {})
        binary = evaluation.get("binary", {})
        subtype = evaluation.get("subtype", {})
        if not isinstance(binary, dict) or not isinstance(subtype, dict):
            continue
        return {
            "seed": seed,
            "status": "complete",
            "run_id": str(metrics.get("run_dir", run)),
            "binary_auroc": float(binary["AUROC"]),
            "subtype_macro_f1": float(subtype["macro_f1"]),
        }
    return None


def _balanced_experiment(root: Path) -> dict[str, Any]:
    """Return the newest complete full class-balanced experiment, if present."""

    candidates = sorted(
        (root / "experiments" / "btxrd_subtype_balanced").glob("run-*/contract.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for contract_path in candidates:
        run = contract_path.parent
        metrics_path = run / "metrics.json"
        status_path = run / "status.json"
        if not metrics_path.is_file() or not status_path.is_file():
            continue
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        execution = contract.get("execution", {})
        if (
            status.get("status") != "complete"
            or execution.get("max_batches") is not None
        ):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        evaluation = metrics.get("evaluation", {})
        binary = evaluation.get("binary", {})
        subtype = evaluation.get("subtype", {})
        if not isinstance(binary, dict) or not isinstance(subtype, dict):
            continue
        return {
            "status": "complete",
            "run_id": str(metrics.get("run_dir", run)),
            "binary_auroc": float(binary["AUROC"]),
            "subtype_macro_f1": float(subtype["macro_f1"]),
            "subtype_balanced_accuracy": float(subtype["balanced_accuracy"]),
            "benchmark_eligible": execution.get("epochs_override") is None,
            "epochs_override": execution.get("epochs_override"),
            "class_weights": contract.get("runtime", {}).get("training", {}).get(
                "subtype_class_weights", []
            ),
        }
    return {"status": "not_run", "metrics": "Not run"}


def build_robustness_report(
    project_root: Path,
    *,
    seeds: Sequence[int] = (42, 43, 44),
    external_root: Path | None = None,
) -> dict[str, Any]:
    """Report executed seeds and leave unexecuted/external work explicitly pending."""

    root = Path(project_root).resolve()
    seed_results = []
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("seeds must be non-negative integers")
        completed = _completed_seed(root, seed)
        seed_results.append(
            completed
            if completed is not None
            else {"seed": seed, "status": "not_run", "metrics": "Not run"}
        )
    if external_root is None:
        external = {"status": "unavailable", "reason": "No external dataset supplied"}
    elif not Path(external_root).is_dir():
        external = {"status": "unavailable", "reason": "External dataset path is missing"}
    else:
        external = {
            "status": "pending",
            "reason": "External labels/splits require a verified dataset contract",
            "path": str(Path(external_root).resolve()),
        }
    completed = [item for item in seed_results if item.get("status") == "complete"]
    if len(completed) >= 2:
        binary_values = [float(item["binary_auroc"]) for item in completed]
        subtype_values = [float(item["subtype_macro_f1"]) for item in completed]
        aggregate: dict[str, Any] = {
            "status": "complete",
            "completed_seed_count": len(completed),
            "binary_auroc_mean": sum(binary_values) / len(binary_values),
            "binary_auroc_std": _population_std(binary_values),
            "subtype_macro_f1_mean": sum(subtype_values) / len(subtype_values),
            "subtype_macro_f1_std": _population_std(subtype_values),
        }
    else:
        aggregate = {
            "status": "pending",
            "completed_seed_count": len(completed),
            "metrics": "Not run",
            "reason": "At least two complete full-seed runs are required",
        }
    return {
        "protocol": "btxrd_robustness_bookkeeping",
        "seed_results": seed_results,
        "aggregate": aggregate,
        "balanced_experiment": _balanced_experiment(root),
        "external_validation": external,
        "no_fabricated_metrics": True,
    }


def _population_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


__all__ = ["build_robustness_report"]
