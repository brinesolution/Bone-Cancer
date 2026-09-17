"""One-time frozen test evaluation and explainability workflow for Phase 9."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bone_cancer_ai.artifacts.registry import (
    CheckpointRecord,
    checkpoint_publication_lock,
    resolve_checkpoint,
    update_checkpoint_test_status,
)
from bone_cancer_ai.data.datamodule import BoneCancerDataModule
from bone_cancer_ai.evaluation.bootstrap import bootstrap_confidence_intervals
from bone_cancer_ai.evaluation.calibration import binary_entropy
from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.evaluation.comparison import (
    ensure_matching_cohort,
    plot_final_comparison,
    validate_prediction_frame,
)
from bone_cancer_ai.evaluation.reports import (
    atomic_write_csv,
    atomic_write_json,
    measure_inference_latency,
)
from bone_cancer_ai.explainability.gallery import select_gallery_cases
from bone_cancer_ai.explainability.gradcam import GradCAM
from bone_cancer_ai.explainability.overlays import denormalize_image, overlay_heatmap
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator
from bone_cancer_ai.optimization.contracts import load_yaml, sha256_file
from bone_cancer_ai.optimization.frozen import load_frozen_phase8_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PHASE9_PROTOCOL = "phase9_frozen_test_v1"
TEST_PREDICTIONS_NAME = "hosp_phase9_test_predictions.csv"
TEST_ATTEMPT_NAME = "hosp_phase9_test_attempt.json"
TEST_LEDGER_NAME = "hosp_phase9_test_ledger.json"


@dataclass(frozen=True)
class SealedPredictions:
    frame: pd.DataFrame
    reused: bool
    ledger: dict[str, Any]


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _expected_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    expected = frame.loc[:, ["sample_id", "binary_label"]].copy()
    expected.columns = ["sample_id", "target"]
    expected["sample_id"] = expected["sample_id"].astype(str)
    expected["target"] = expected["target"].astype(int)
    if expected.empty or expected["sample_id"].duplicated().any():
        raise ValueError("expected test cohort is empty or has duplicate IDs")
    return expected.sort_values("sample_id", kind="stable").reset_index(drop=True)


def _matches_expected(predictions: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    checked = validate_prediction_frame(predictions, label="HOSP-BoneNet")
    if not checked["sample_id"].equals(expected["sample_id"]):
        raise ValueError("HOSP test predictions do not match the sealed test sample IDs")
    if not checked["target"].astype(int).equals(expected["target"].astype(int)):
        raise ValueError("HOSP test prediction targets do not match the test manifest")
    return checked


def seal_or_reuse_test_predictions(
    artifact_root: Path,
    *,
    contract: Mapping[str, Any],
    expected: pd.DataFrame,
    predictor: Callable[[], pd.DataFrame],
) -> SealedPredictions:
    """Run test inference at most once, recovering only checksum-valid completed output."""
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    predictions_path = root / TEST_PREDICTIONS_NAME
    attempt_path = root / TEST_ATTEMPT_NAME
    ledger_path = root / TEST_LEDGER_NAME
    strict_contract = json.loads(json.dumps(dict(contract), sort_keys=True, allow_nan=False))
    contract_digest = _canonical_digest(strict_contract)

    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("status") != "complete" or ledger.get("contract") != strict_contract:
            raise ValueError("sealed HOSP test ledger does not match the evaluation contract")
        if not predictions_path.is_file() or sha256_file(predictions_path) != ledger.get(
            "predictions_sha256"
        ):
            raise ValueError("sealed HOSP test predictions are missing or checksum-invalid")
        frame = _matches_expected(pd.read_csv(predictions_path), expected)
        if int(ledger.get("sample_count", -1)) != len(frame):
            raise ValueError("sealed HOSP test sample count is invalid")
        return SealedPredictions(frame=frame, reused=True, ledger=ledger)

    if predictions_path.exists() and not attempt_path.exists():
        raise ValueError("unbound HOSP test predictions exist without an attempt contract")
    if attempt_path.exists():
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt.get("contract") != strict_contract or attempt.get(
            "contract_sha256"
        ) != contract_digest:
            raise ValueError("an earlier HOSP test attempt used a different contract")
        if predictions_path.exists():
            frame = _matches_expected(pd.read_csv(predictions_path), expected)
            ledger = {
                "status": "complete",
                "contract": strict_contract,
                "contract_sha256": contract_digest,
                "predictions_sha256": sha256_file(predictions_path),
                "sample_count": len(frame),
                "evaluation_count": 1,
                "recovered_after_interruption": True,
            }
            atomic_write_json(ledger, ledger_path)
            atomic_write_json({**attempt, "status": "complete"}, attempt_path)
            return SealedPredictions(frame=frame, reused=True, ledger=ledger)
        raise RuntimeError(
            "a matching HOSP test attempt is incomplete; automatic repeat inference is blocked"
        )
    else:
        atomic_write_json(
            {
                "status": "in_progress",
                "contract": strict_contract,
                "contract_sha256": contract_digest,
            },
            attempt_path,
        )

    frame = _matches_expected(predictor(), expected)
    atomic_write_csv(frame, predictions_path)
    ledger = {
        "status": "complete",
        "contract": strict_contract,
        "contract_sha256": contract_digest,
        "predictions_sha256": sha256_file(predictions_path),
        "sample_count": len(frame),
        "evaluation_count": 1,
        "recovered_after_interruption": False,
    }
    atomic_write_json(ledger, ledger_path)
    atomic_write_json(
        {"status": "complete", "contract": strict_contract, "contract_sha256": contract_digest},
        attempt_path,
    )
    return SealedPredictions(frame=frame, reused=False, ledger=ledger)


def predict_binary_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> pd.DataFrame:
    """Run ordered valid-sample inference and return the standard prediction schema."""
    sample_ids: list[str] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rejected = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            valid = (
                batch["valid_task_mask"]["sample"].bool()
                & batch["valid_task_mask"]["binary"].bool()
            )
            rejected += int((~valid).sum())
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(images)
                batch_logits = output["binary_logits"].reshape(-1)
            if valid.any():
                logits.append(batch_logits.detach().cpu()[valid])
                targets.append(batch["binary_label"].detach().cpu()[valid].to(torch.long))
                ids = list(batch["metadata"]["sample_id"])
                sample_ids.extend(
                    str(value)
                    for value, keep in zip(ids, valid.tolist(), strict=True)
                    if keep
                )
    if rejected:
        raise RuntimeError(f"frozen inference rejected {rejected} audited samples")
    if not logits:
        raise RuntimeError("frozen inference produced no valid samples")
    logit = torch.cat(logits).to(torch.float64)
    target = torch.cat(targets).to(torch.long)
    probability = torch.sigmoid(logit)
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target": target.numpy(),
            "logit": logit.numpy(),
            "probability": probability.numpy(),
            "prediction": (probability >= 0.5).to(torch.long).numpy(),
        }
    )
    return validate_prediction_frame(frame, label="frozen inference")


def _project_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes the project root")
    return resolved


def _load_configuration(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    required = {
        "phase",
        "output_root",
        "hosp_alias",
        "baseline_aliases",
        "data_config",
        "schema_path",
        "split_dir",
        "device",
        "amp",
        "threshold",
        "bootstrap",
        "conformal",
        "gallery",
    }
    missing = required - set(config)
    if missing or config.get("phase") != 9:
        raise ValueError(f"invalid Phase 9 configuration; missing={sorted(missing)}")
    aliases = config["baseline_aliases"]
    if not isinstance(aliases, dict) or set(aliases) != {
        "DenseNet121",
        "EfficientNet-B3",
        "ConvNeXt-Tiny",
        "Swin-Tiny",
    }:
        raise ValueError("baseline_aliases must define the four required comparison models")
    if float(config["threshold"]) != 0.5:
        raise ValueError("Phase 9 must retain the frozen binary threshold 0.5")
    return config


def _make_data_module(root: Path, config: Mapping[str, Any]) -> BoneCancerDataModule:
    data_path = _project_path(root, config["data_config"], name="data_config")
    schema_path = _project_path(root, config["schema_path"], name="schema_path")
    split_dir = _project_path(root, config["split_dir"], name="split_dir")
    data = deepcopy(load_yaml(data_path))
    data["loader"] = deepcopy(data.get("loader", {}))
    data["loader"].update(
        {
            "batch_size": int(config.get("batch_size", 8)),
            "num_workers": int(config.get("num_workers", 0)),
            "pin_memory": True,
            "persistent_workers": False,
            "sampling": "shuffle",
        }
    )
    module = BoneCancerDataModule(
        split_dir, data, load_yaml(schema_path), splits=("calibration", "test")
    )
    module.setup()
    return module


def _verify_metric_record(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any], *, label: str
) -> None:
    for metric, value in recomputed.items():
        if metric == "confusion_matrix":
            if recorded.get(metric) != value:
                raise ValueError(f"{label} recorded confusion matrix is stale")
        elif value is None:
            if recorded.get(metric) is not None:
                raise ValueError(f"{label} recorded {metric} is stale")
        elif not math.isclose(
            float(recorded.get(metric)), float(value), rel_tol=1e-8, abs_tol=1e-8
        ):
            raise ValueError(f"{label} recorded {metric} is stale")


def _load_baseline_predictions(
    root: Path,
    aliases: Mapping[str, str],
    expected: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, CheckpointRecord]]:
    predictions: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    records: dict[str, CheckpointRecord] = {}
    for name, alias in aliases.items():
        record = resolve_checkpoint(root, str(alias))
        if record.test_status.get("status") != "complete" or record.test_status.get("phase") != 5:
            raise ValueError(f"{name} does not have a completed Phase 5 test record")
        run_dir = _project_path(root, record.source_run, name=f"{name} source_run")
        prediction_path = run_dir / "test_predictions.csv"
        metrics_path = run_dir / "metrics.json"
        frame = _matches_expected(pd.read_csv(prediction_path), expected)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recomputed = binary_classification_metrics(frame["logit"], frame["target"])
        _verify_metric_record(metrics["test"], recomputed, label=name)
        predictions[name] = frame
        metadata[name] = {
            "metrics": recomputed,
            "params": int(metrics["resources"]["trainable_parameters"]),
            "latency_ms": float(metrics["resources"]["latency"]["latency_ms"]),
            "prediction_path": prediction_path,
            "prediction_sha256": sha256_file(prediction_path),
        }
        records[name] = record
    ensure_matching_cohort(predictions)
    return predictions, metadata, records


def _comparison_row(
    *,
    name: str,
    alias: str,
    checkpoint_sha256: str,
    predictions: pd.DataFrame,
    params: int,
    latency_ms: float,
    prediction_sha256: str,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = binary_classification_metrics(predictions["logit"], predictions["target"])
    intervals = bootstrap_confidence_intervals(
        predictions["logit"],
        predictions["target"],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    row: dict[str, Any] = {
        "model": name,
        "checkpoint_alias": alias,
        "checkpoint_sha256": checkpoint_sha256,
        "test_samples": len(predictions),
    }
    for metric in (
        "AUROC",
        "AUPRC",
        "balanced_accuracy",
        "F1",
        "sensitivity",
        "specificity",
        "MCC",
        "Brier",
        "ECE",
    ):
        row[metric] = metrics[metric]
    for metric, interval in intervals.items():
        row[f"{metric}_ci_lower"] = interval["lower"]
        row[f"{metric}_ci_upper"] = interval["upper"]
    row.update(
        {
            "bootstrap_confidence": confidence,
            "bootstrap_resamples": resamples,
            "params": params,
            "latency_ms": latency_ms,
            "test_prediction_sha256": prediction_sha256,
        }
    )
    return row, {"metrics": metrics, "intervals": intervals}


def _save_gallery(
    records: list[dict[str, Any]], destination: Path, *, title: str
) -> Path:
    if not records:
        raise ValueError("cannot render an empty explanation gallery")
    figure, axes = plt.subplots(len(records), 3, figsize=(12, 3.6 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        for column, key in enumerate(("image", "lesion_overlay", "gradcam_overlay")):
            axes[row, column].imshow(record[key], cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[row, 0].set_title(record["title"], fontsize=8)
        axes[row, 1].set_title("Unverified lesion-head spatial attention", fontsize=8)
        axes[row, 2].set_title("Grad-CAM (predicted class)", fontsize=8)
    figure.suptitle(title, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), h_pad=2.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    try:
        figure.savefig(temporary, dpi=140, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return destination


def _generate_explanations(
    model: torch.nn.Module,
    module: BoneCancerDataModule,
    predictions: pd.DataFrame,
    *,
    device: torch.device,
    output_root: Path,
    per_category: int,
) -> dict[str, Any]:
    selected = select_gallery_cases(predictions, per_category=per_category)
    dataset = module.datasets["test"]
    positions = {
        str(sample_id): index
        for index, sample_id in enumerate(dataset.frame["sample_id"].astype(str))
    }
    records_by_category: dict[str, list[dict[str, Any]]] = {}
    index_rows: list[dict[str, Any]] = []
    target_layer = model.feature_projection
    with GradCAM(model, target_layer) as explainer:
        for row in selected.to_dict(orient="records"):
            sample_id = str(row["sample_id"])
            item = dataset[positions[sample_id]]
            image_tensor = item["image"].unsqueeze(0).to(device)
            predicted = int(row["prediction"])
            gradcam = explainer.generate(
                image_tensor, target_classes=torch.tensor([predicted], device=device)
            )[0].cpu().numpy()
            with torch.inference_mode():
                output = model(image_tensor)
            if "lesion_logits" not in output:
                raise ValueError("selected HOSP configuration lacks lesion logits")
            lesion = output["lesion_logits"][0, 0].sigmoid().detach().cpu().numpy()
            image = denormalize_image(image_tensor[0])
            lesion_overlay = overlay_heatmap(image, lesion)
            gradcam_overlay = overlay_heatmap(image, gradcam)
            conformal_set = str(row["conformal_set"])
            confidence = max(float(row["probability"]), 1.0 - float(row["probability"]))
            title = (
                f"{sample_id} | true={int(row['target'])} pred={predicted} "
                f"p={float(row['probability']):.3f} conf={confidence:.3f} "
                f"H={float(row['uncertainty']):.3f} "
                f"set={conformal_set}\nsubtype=Unavailable concepts=Unavailable"
            )
            record = {
                "sample_id": sample_id,
                "category": str(row["category"]),
                "image": image,
                "lesion_overlay": lesion_overlay,
                "gradcam_overlay": gradcam_overlay,
                "title": title,
            }
            records_by_category.setdefault(record["category"], []).append(record)
            index_rows.append(
                {
                    "category": record["category"],
                    "sample_id": sample_id,
                    "target": int(row["target"]),
                    "prediction": predicted,
                    "probability": float(row["probability"]),
                    "confidence": confidence,
                    "uncertainty": float(row["uncertainty"]),
                    "conformal_set": conformal_set,
                    "subtype": "Unavailable",
                    "concept_maps": "Unavailable",
                }
            )
    galleries: dict[str, str] = {}
    for category, records in records_by_category.items():
        path = _save_gallery(
            records,
            output_root / category / "gallery.png",
            title=category.replace("_", " ").title(),
        )
        galleries[category] = path.as_posix()
    failures = records_by_category.get("false_positive", []) + records_by_category.get(
        "false_negative", []
    )
    if not failures:
        raise ValueError("test predictions produced no failure cases for the required gallery")
    failure_path = _save_gallery(
        failures,
        output_root / "failure_gallery.png",
        title="HOSP-BoneNet failure gallery",
    )
    index_path = output_root / "gallery_index.csv"
    atomic_write_csv(pd.DataFrame(index_rows), index_path)
    qa = {
        "status": "pending_visual_inspection",
        "sample_count": len(index_rows),
        "categories": {key: len(value) for key, value in records_by_category.items()},
        "finite_maps": True,
        "aligned_output_shape": [384, 384],
        "lesion_map_claim": "unverified spatial attention; no probability/segmentation claim",
        "concept_maps": "Unavailable: concept labels/head disabled",
        "subtype": "Unavailable: subtype labels/head disabled",
        "open_set": "Unavailable: no verified OOD evaluation labels",
        "manual_checks": [
            "text-marker shortcut",
            "crop or padding error",
            "mislabeled overlay",
            "gross localization outside visible anatomy",
            "broken map",
            "inverted image",
        ],
    }
    qa_path = output_root / "visual_qa.json"
    atomic_write_json(qa, qa_path)
    return {
        "galleries": galleries,
        "failure_gallery": failure_path.as_posix(),
        "index": index_path.as_posix(),
        "visual_qa": qa_path.as_posix(),
    }


def _status_artifacts(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.resolve().relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(set(paths))
    }


def _validate_visual_qa(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("Phase 9 visual QA record is missing")
    qa = json.loads(path.read_text(encoding="utf-8"))
    if qa.get("status") not in {"complete", "complete_with_limitations"}:
        raise ValueError("Phase 9 visual QA has not been completed")
    if not isinstance(qa.get("review_type"), str) or not qa["review_type"].strip():
        raise ValueError("Phase 9 visual QA review type is missing")
    if not isinstance(qa.get("checks"), dict) or not qa["checks"]:
        raise ValueError("Phase 9 visual QA checks are missing")
    if not isinstance(qa.get("conclusion"), str) or not qa["conclusion"].strip():
        raise ValueError("Phase 9 visual QA conclusion is missing")
    return qa


def _validate_run_artifacts(
    root: Path,
    status: Mapping[str, Any],
    *,
    ignored: set[str] | None = None,
) -> None:
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Phase 9 artifact manifest is empty")
    ignored = ignored or set()
    for relative, digest in artifacts.items():
        if relative in ignored:
            continue
        path = _project_path(root, relative, name="Phase 9 artifact")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Phase 9 artifact is missing or changed: {relative}")


def _validate_completed_run(
    root: Path,
    run_dir: Path,
    *,
    expected_contract: Mapping[str, Any],
    expected_fingerprint: str,
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise ValueError("completed Phase 9 run has no status record")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("protocol") != PHASE9_PROTOCOL:
        raise ValueError("Phase 9 status record is incomplete")
    if status.get("contract") != dict(expected_contract):
        raise ValueError("completed Phase 9 run does not match the requested contract")
    if status.get("fingerprint") != expected_fingerprint:
        raise ValueError("completed Phase 9 run fingerprint does not match the requested contract")
    _validate_run_artifacts(root, status)
    qa_path = root / "artifacts" / "explainability" / "phase9" / "visual_qa.json"
    _validate_visual_qa(qa_path)
    return status


def _evaluation_inputs(
    root: Path,
    config: Mapping[str, Any],
    record: CheckpointRecord,
) -> tuple[
    BoneCancerDataModule,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
    dict[str, CheckpointRecord],
    dict[str, Any],
    str,
]:
    module = _make_data_module(root, config)
    expected = _expected_cohort(module.frames["test"])
    baseline_predictions, baseline_metadata, baseline_records = _load_baseline_predictions(
        root, config["baseline_aliases"], expected
    )
    contract = {
        "protocol": PHASE9_PROTOCOL,
        "checkpoint_alias": record.alias,
        "checkpoint_sha256": record.checkpoint_sha256,
        "configuration_fingerprint": record.configuration_fingerprint,
        "data_config_sha256": sha256_file(
            _project_path(root, config["data_config"], name="data_config")
        ),
        "test_manifest_sha256": sha256_file(
            _project_path(root, config["split_dir"], name="split_dir") / "test.csv"
        ),
        "test_cohort_sha256": _canonical_digest(expected.to_dict(orient="list")),
        "sample_count": len(expected),
        "threshold": float(config["threshold"]),
    }
    payload = {
        "contract": contract,
        "evaluation_config": json.loads(
            json.dumps(dict(config), sort_keys=True, allow_nan=False)
        ),
        "schema_sha256": sha256_file(
            _project_path(root, config["schema_path"], name="schema_path")
        ),
        "calibration_manifest_sha256": sha256_file(
            _project_path(root, config["split_dir"], name="split_dir") / "calibration.csv"
        ),
        "baseline_prediction_sha256": {
            name: metadata["prediction_sha256"] for name, metadata in baseline_metadata.items()
        },
        "bootstrap": config["bootstrap"],
        "conformal": config["conformal"],
        "gallery": config["gallery"],
    }
    return (
        module,
        expected,
        baseline_predictions,
        baseline_metadata,
        baseline_records,
        payload,
        _canonical_digest(payload),
    )


def _validate_test_binding(root: Path, record: CheckpointRecord) -> dict[str, Any]:
    ledger_path = root / "artifacts" / "metrics" / TEST_LEDGER_NAME
    predictions_path = root / "artifacts" / "metrics" / TEST_PREDICTIONS_NAME
    if not ledger_path.is_file() or not predictions_path.is_file():
        raise ValueError("Phase 9 test ledger or predictions are missing")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    contract = ledger.get("contract")
    digest = ledger.get("predictions_sha256")
    if (
        ledger.get("status") != "complete"
        or ledger.get("evaluation_count") != 1
        or not isinstance(contract, dict)
        or contract.get("checkpoint_alias") != record.alias
        or contract.get("checkpoint_sha256") != record.checkpoint_sha256
        or sha256_file(predictions_path) != digest
    ):
        raise ValueError("Phase 9 test ledger is stale or bound to another checkpoint")
    registered_digest = record.test_status.get("predictions_sha256")
    if registered_digest is not None and registered_digest != digest:
        raise ValueError("checkpoint registry and Phase 9 test ledger disagree")
    return ledger


def _repair_registry_status(
    root: Path, record: CheckpointRecord, run_dir: Path, ledger: Mapping[str, Any]
) -> None:
    desired = {
        "status": "complete",
        "phase": 9,
        "evaluation_run": run_dir.resolve().relative_to(root).as_posix(),
        "predictions_sha256": str(ledger["predictions_sha256"]),
    }
    if record.test_status != desired:
        update_checkpoint_test_status(
            root,
            alias=record.alias,
            expected_checkpoint_sha256=record.checkpoint_sha256,
            test_status=desired,
        )


def _run_phase9_locked(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_configuration(root, config_path)
    alias = str(config["hosp_alias"])
    record = resolve_checkpoint(root, alias)
    if record.model_family != "hosp_bonenet" or record.phase != "phase_8":
        raise ValueError("hosp_alias must resolve to the frozen Phase 8 HOSP checkpoint")
    (
        module,
        expected,
        baseline_predictions,
        baseline_metadata,
        baseline_records,
        fingerprint_payload,
        fingerprint,
    ) = _evaluation_inputs(root, config, record)
    output_root = _project_path(root, config["output_root"], name="output_root")
    run_dir = output_root / f"phase9-{fingerprint[:12]}"
    if record.test_status.get("status") == "complete":
        existing = _project_path(root, record.test_status["evaluation_run"], name="evaluation_run")
        if existing != run_dir:
            raise ValueError("completed Phase 9 evaluation does not match the requested contract")
        status = _validate_completed_run(
            root,
            existing,
            expected_contract=fingerprint_payload,
            expected_fingerprint=fingerprint,
        )
        _validate_test_binding(root, record)
        return {**status["summary"], "run_dir": existing.as_posix(), "reused": True}
    if record.test_status != {"status": "not_run", "planned_phase": 9}:
        raise ValueError("HOSP checkpoint has an invalid pre-evaluation test status")

    metrics_root = root / "artifacts" / "metrics"
    figures_root = root / "artifacts" / "figures"
    explainability_root = root / "artifacts" / "explainability" / "phase9"
    contract = fingerprint_payload["contract"]
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            status = _validate_completed_run(
                root,
                run_dir,
                expected_contract=fingerprint_payload,
                expected_fingerprint=fingerprint,
            )
            ledger = _validate_test_binding(root, record)
            _repair_registry_status(root, record, run_dir, ledger)
            return {**status["summary"], "run_dir": run_dir.as_posix(), "reused": True}
        if status.get("status") == "awaiting_visual_qa":
            if status.get("contract") != fingerprint_payload:
                raise ValueError("pending Phase 9 run does not match the requested contract")
            _validate_run_artifacts(root, status)
            return {**status["summary"], "run_dir": run_dir.as_posix(), "reused": True}
    atomic_write_json(
        {
            "status": "running",
            "protocol": PHASE9_PROTOCOL,
            "fingerprint": fingerprint,
            "contract": fingerprint_payload,
        },
        status_path,
    )

    requested_device = torch.device(str(config["device"]))
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 9 requires CUDA but CUDA is unavailable")
    model, _frozen, loaded_record = load_frozen_phase8_checkpoint(
        root, alias, device=requested_device
    )
    if loaded_record != record:
        raise ValueError("frozen checkpoint record changed during Phase 9 startup")
    amp = bool(config["amp"])
    calibration_predictions = predict_binary_model(
        model, module.loader("calibration"), device=requested_device, amp=amp
    )
    calibration_path = run_dir / "calibration_predictions.csv"
    atomic_write_csv(calibration_predictions, calibration_path)

    sealed = seal_or_reuse_test_predictions(
        metrics_root,
        contract=contract,
        expected=expected,
        predictor=partial(
            predict_binary_model,
            model,
            module.loader("test"),
            device=requested_device,
            amp=amp,
        ),
    )
    conformal = BinaryConformalCalibrator.fit(
        calibration_predictions["probability"],
        calibration_predictions["target"],
        alpha=float(config["conformal"]["alpha"]),
    )
    enriched = sealed.frame.copy()
    sets = conformal.predict_sets(enriched["probability"])
    enriched["uncertainty"] = binary_entropy(enriched["probability"])
    enriched["conformal_set"] = [json.dumps(values) for values in sets]
    enriched["conformal_size"] = [len(values) for values in sets]
    enriched["rejected"] = enriched["conformal_size"] != 1
    enriched_path = run_dir / "hosp_test_predictions_enriched.csv"
    atomic_write_csv(enriched, enriched_path)
    conformal_summary = conformal.summarize(enriched["probability"], enriched["target"])
    conformal_path = run_dir / "conformal_summary.json"
    atomic_write_json(conformal_summary, conformal_path)
    conformal_artifact_path = metrics_root / "hosp_conformal_calibration.json"
    atomic_write_json(conformal_summary, conformal_artifact_path)

    example = next(iter(module.loader("test")))["image"][:1]
    latency = measure_inference_latency(
        model,
        example,
        requested_device,
        warmup=int(config.get("latency_warmup", 10)),
        repetitions=int(config.get("latency_repetitions", 50)),
    )
    model_params = sum(parameter.numel() for parameter in model.parameters())
    all_predictions = {**baseline_predictions, "HOSP-BoneNet": sealed.frame}
    ensure_matching_cohort(all_predictions)
    bootstrap = config["bootstrap"]
    rows: list[dict[str, Any]] = []
    statistical_report: dict[str, Any] = {}
    for name, frame in all_predictions.items():
        if name == "HOSP-BoneNet":
            row, report = _comparison_row(
                name=name,
                alias=alias,
                checkpoint_sha256=record.checkpoint_sha256,
                predictions=frame,
                params=model_params,
                latency_ms=float(latency["latency_ms"]),
                prediction_sha256=str(sealed.ledger["predictions_sha256"]),
                resamples=int(bootstrap["resamples"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]),
            )
        else:
            metadata = baseline_metadata[name]
            baseline_record = baseline_records[name]
            row, report = _comparison_row(
                name=name,
                alias=str(config["baseline_aliases"][name]),
                checkpoint_sha256=baseline_record.checkpoint_sha256,
                predictions=frame,
                params=int(metadata["params"]),
                latency_ms=float(metadata["latency_ms"]),
                prediction_sha256=str(metadata["prediction_sha256"]),
                resamples=int(bootstrap["resamples"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]),
            )
        rows.append(row)
        statistical_report[name] = report
    comparison = pd.DataFrame(rows)
    comparison_path = metrics_root / "final_model_comparison.csv"
    atomic_write_csv(comparison, comparison_path)
    intervals_path = metrics_root / "final_model_intervals.json"
    atomic_write_json(statistical_report, intervals_path)
    figures = plot_final_comparison(all_predictions, figures_root)

    explanations = _generate_explanations(
        model,
        module,
        enriched,
        device=requested_device,
        output_root=explainability_root,
        per_category=int(config["gallery"]["per_category"]),
    )
    del model
    if requested_device.type == "cuda":
        torch.cuda.empty_cache()

    final_config_path = run_dir / "resolved_config.yaml"
    shutil.copyfile(config_path, final_config_path)
    artifact_paths = [
        comparison_path,
        intervals_path,
        metrics_root / TEST_PREDICTIONS_NAME,
        metrics_root / TEST_LEDGER_NAME,
        Path(figures["roc"]),
        Path(figures["pr"]),
        Path(figures["calibration"]),
        *[Path(value) for value in figures["confusion_matrices"]],
        Path(explanations["failure_gallery"]),
        *[Path(value) for value in explanations["galleries"].values()],
        Path(explanations["index"]),
        Path(explanations["visual_qa"]),
        calibration_path,
        enriched_path,
        conformal_path,
        conformal_artifact_path,
        final_config_path,
    ]
    summary = {
        "status": "awaiting_visual_qa",
        "fingerprint": fingerprint,
        "checkpoint_alias": record.alias,
        "checkpoint_sha256": record.checkpoint_sha256,
        "hosp_test_metrics": statistical_report["HOSP-BoneNet"]["metrics"],
        "conformal": conformal_summary,
        "test_inference_reused": sealed.reused,
        "test_evaluation_count": 1,
        "test_samples": len(sealed.frame),
        "comparison_models": list(all_predictions),
        "open_set_status": "Unavailable: no verified OOD labels",
        "concept_status": "Unavailable: concept labels/head disabled",
        "subtype_status": "Unavailable: subtype labels/head disabled",
        "visual_qa_status": "pending_visual_inspection",
    }
    status = {
        "status": "awaiting_visual_qa",
        "protocol": PHASE9_PROTOCOL,
        "fingerprint": fingerprint,
        "contract": fingerprint_payload,
        "summary": summary,
        "artifacts": _status_artifacts(root, artifact_paths),
    }
    atomic_write_json(status, status_path)
    return {**summary, "run_dir": run_dir.as_posix(), "reused": False}


def _finalize_phase9_visual_qa_locked(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = _load_configuration(root, Path(config_path).resolve())
    record = resolve_checkpoint(root, str(config["hosp_alias"]))
    if record.model_family != "hosp_bonenet" or record.phase != "phase_8":
        raise ValueError("hosp_alias must resolve to the frozen Phase 8 HOSP checkpoint")
    (
        _module,
        _expected,
        _baseline_predictions,
        _baseline_metadata,
        _baseline_records,
        fingerprint_payload,
        fingerprint,
    ) = _evaluation_inputs(root, config, record)
    run_dir = _project_path(root, config["output_root"], name="output_root") / (
        f"phase9-{fingerprint[:12]}"
    )
    status_path = run_dir / "status.json"
    if record.test_status.get("status") == "complete":
        status = _validate_completed_run(
            root,
            run_dir,
            expected_contract=fingerprint_payload,
            expected_fingerprint=fingerprint,
        )
        _validate_test_binding(root, record)
        return {**status["summary"], "run_dir": run_dir.as_posix(), "reused": True}
    if record.test_status != {"status": "not_run", "planned_phase": 9}:
        raise ValueError("HOSP checkpoint has an invalid visual-QA publication status")
    if not status_path.is_file():
        raise ValueError("Phase 9 evaluation must run before visual-QA finalization")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "awaiting_visual_qa":
        raise ValueError("Phase 9 run is not awaiting visual QA")
    if status.get("contract") != fingerprint_payload or status.get("fingerprint") != fingerprint:
        raise ValueError("pending Phase 9 run does not match the requested contract")
    qa_path = root / "artifacts" / "explainability" / "phase9" / "visual_qa.json"
    qa_relative = qa_path.relative_to(root).as_posix()
    _validate_run_artifacts(root, status, ignored={qa_relative})
    qa = _validate_visual_qa(qa_path)
    artifacts = dict(status["artifacts"])
    artifacts[qa_relative] = sha256_file(qa_path)
    summary = dict(status["summary"])
    summary.update({"status": "complete", "visual_qa_status": qa["status"]})
    completed = {
        **status,
        "status": "complete",
        "summary": summary,
        "artifacts": artifacts,
    }
    atomic_write_json(completed, status_path)
    completed = _validate_completed_run(
        root,
        run_dir,
        expected_contract=fingerprint_payload,
        expected_fingerprint=fingerprint,
    )
    ledger = _validate_test_binding(root, record)
    _repair_registry_status(root, record, run_dir, ledger)
    return {**completed["summary"], "run_dir": run_dir.as_posix(), "reused": False}


def run_phase9(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Execute or safely reuse Phase 9 under a process-wide checkpoint lock."""
    root = Path(project_root).resolve()
    config = _load_configuration(root, Path(config_path).resolve())
    with checkpoint_publication_lock(root, str(config["hosp_alias"])):
        return _run_phase9_locked(root, config_path)


def finalize_phase9_visual_qa(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Publish a completed Phase 9 run after its visual-QA record is accepted."""
    root = Path(project_root).resolve()
    config = _load_configuration(root, Path(config_path).resolve())
    with checkpoint_publication_lock(root, str(config["hosp_alias"])):
        return _finalize_phase9_visual_qa_locked(root, config_path)


__all__ = [
    "PHASE9_PROTOCOL",
    "SealedPredictions",
    "finalize_phase9_visual_qa",
    "predict_binary_model",
    "run_phase9",
    "seal_or_reuse_test_predictions",
]
