"""Train one approved Phase 5 baseline or validate completed run bundles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import random
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_MODELS = ("densenet121", "efficientnet_b3", "convnext_tiny", "swin_tiny")
BASELINE_SUMMARY_COLUMNS = [
    "model",
    "seed",
    "best_epoch",
    "AUROC",
    "AUPRC",
    "balanced_accuracy",
    "F1",
    "sensitivity",
    "specificity",
    "MCC",
    "params",
    "latency_ms",
]
REQUIRED_RUN_FILES = {
    "config.yaml",
    "environment.txt",
    "git_commit.txt",
    "metrics.json",
    "history.csv",
    "best.pt",
    "last.pt",
    "best_checkpoint_path.txt",
    "confusion_matrix.csv",
    "notes.md",
    "test_predictions.csv",
}
COMMON_CONFIG_KEYS = {
    "data_config",
    "output_root",
    "image_size",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "warmup_epochs",
    "amp",
    "gradient_accumulation_steps",
    "gradient_clip_norm",
    "early_stopping_patience",
    "seed",
    "device",
    "allow_cpu_training",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "selection_split",
    "selection_metric",
}
DATA_CONFIG_KEYS = {
    "project_root",
    "dataset_root",
    "source_read_only",
    "workspace_root",
    "artifacts_root",
    "logs_root",
    "schema_path",
    "official_splits",
    "image_extensions",
    "image_size",
    "seed",
    "labels_verified",
    "audit",
    "provenance",
    "split_policy",
    "preprocessing",
    "augmentation",
    "loader",
}
FINAL_POLICY = {
    "epochs": 40,
    "early_stopping_patience": 8,
    "image_size": 384,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "warmup_epochs": 2,
    "amp": True,
    "gradient_clip_norm": 1.0,
    "seed": 42,
    "device": "cuda",
    "allow_cpu_training": False,
    "num_workers": 0,
    "pin_memory": True,
    "persistent_workers": False,
    "pretrained": True,
    "limit_batches": None,
    "selection_split": "val",
    "selection_metric": "AUROC",
}
CONVNEXT_MEMORY_EXCEPTION = {
    "microbatch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "reason": "two batch-8 host-memory allocation failures",
    "evidence_artifact": (
        "artifacts/verification/phase5_convnext_batch8_memory_failures.json"
    ),
    "evidence_sha256": (
        "5803e6397faea3599c5fe75d10eac4948c89ad504efade83c09316a3df87b90d"
    ),
}
MEMORY_EVIDENCE_BUNDLE_NAME = "memory_exception_evidence.json"
MEMORY_EXCEPTION_NOTE_PREFIXES = (
    "Memory exception:",
    "Memory exception reason:",
    "Memory exception evidence:",
    "Memory exception run copy:",
)
PREPROCESSING_POLICY = {
    "preserve_aspect_ratio": True,
    "pad_to_square": True,
    "channels": 3,
    "percentile_clipping": False,
    "force_grayscale": True,
    "pad_value": 0,
    "normalize": "imagenet",
}
AUGMENTATION_POLICY = {
    "horizontal_flip": False,
    "vertical_flip": False,
    "rotation_limit": 7,
    "translate_limit": 0.03,
    "scale_limit": 0.05,
    "brightness_contrast_limit": 0.1,
    "gamma_limit": [90, 110],
    "gaussian_noise_std_range": [0.01, 0.03],
    "clahe": False,
}
LOADER_POLICY = {
    "batch_size": 8,
    "num_workers": 0,
    "pin_memory": True,
    "persistent_workers": False,
    "sampling": "shuffle",
}
EXPECTED_SCHEMA_SHA256 = "6248ff8274f5ad5d26d046f851d672f33390d0a8ef07d6c4b6a0eee11dcefe95"
SPLIT_NAMES = ("fit_train", "calibration", "val", "test")
NESTED_DATA_KEYS = {
    "preprocessing": {
        "preserve_aspect_ratio",
        "pad_to_square",
        "channels",
        "percentile_clipping",
        "force_grayscale",
        "pad_value",
        "normalize",
    },
    "augmentation": {
        "horizontal_flip",
        "vertical_flip",
        "rotation_limit",
        "translate_limit",
        "scale_limit",
        "brightness_contrast_limit",
        "gamma_limit",
        "gaussian_noise_std_range",
        "clahe",
    },
    "loader": {"batch_size", "num_workers", "pin_memory", "persistent_workers", "sampling"},
}
RECORD_CONFIG_KEYS = COMMON_CONFIG_KEYS | {
    "model",
    "project_root",
    "model_config",
    "common_config",
    "dataset_root",
    "split_dir",
    "schema_path",
    "schema_sha256",
    "pretrained",
    "limit_batches",
    "smoke_overrides",
    "data",
    "run_type",
}


@dataclass
class RunDependencies:
    """Inject slow or external boundaries into orchestration tests."""

    data_module_type: type
    trainer_type: type
    create_model: Callable[..., Any]
    count_parameters: Callable[[Any], int]
    class_weight: Callable[[Any], Any]
    load_checkpoint: Callable[..., Mapping[str, Any]]
    metrics: Callable[..., dict[str, object]]
    latency: Callable[..., dict[str, object]]
    cuda_available: Callable[[], bool]
    cuda_peak_memory: Callable[[], int]
    cuda_reset_peak: Callable[[], None]


def _default_dependencies() -> RunDependencies:
    import torch

    from bone_cancer_ai.data.datamodule import BoneCancerDataModule
    from bone_cancer_ai.evaluation.classification import binary_classification_metrics
    from bone_cancer_ai.evaluation.reports import measure_inference_latency
    from bone_cancer_ai.models.baselines.factory import (
        count_trainable_parameters,
        create_baseline,
    )
    from bone_cancer_ai.training.checkpointing import load_checkpoint
    from bone_cancer_ai.training.optimizer import class_weight_from_frame
    from bone_cancer_ai.training.trainer import BinaryTrainer

    return RunDependencies(
        data_module_type=BoneCancerDataModule,
        trainer_type=BinaryTrainer,
        create_model=create_baseline,
        count_parameters=count_trainable_parameters,
        class_weight=class_weight_from_frame,
        load_checkpoint=load_checkpoint,
        metrics=binary_classification_metrics,
        latency=measure_inference_latency,
        cuda_available=torch.cuda.is_available,
        cuda_peak_memory=torch.cuda.max_memory_allocated,
        cuda_reset_peak=torch.cuda.reset_peak_memory_stats,
    )


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a YAML mapping")
    return payload


def _resolve_declared_path(value: object, declaring_file: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured paths must be non-empty strings")
    path = Path(value)
    return (path if path.is_absolute() else declaring_file.parent / path).resolve()


def _require_within(path: Path, root: Path, *, name: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{name} must resolve under project_root")


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _finite_number(config: Mapping[str, Any], key: str, *, positive: bool) -> float:
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be numeric") from error
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{key} must be finite and {qualifier}")
    return number


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() or "."


def _verified_memory_evidence_bytes(
    project_root: Path, memory_exception: Mapping[str, Any]
) -> bytes:
    evidence_path = (project_root / str(memory_exception["evidence_artifact"])).resolve()
    expected_root = (project_root / "artifacts" / "verification").resolve()
    _require_within(evidence_path, expected_root, name="memory exception evidence")
    try:
        contents = evidence_path.read_bytes()
    except OSError as error:
        raise ValueError("memory exception evidence is missing or unreadable") from error
    actual_sha256 = hashlib.sha256(contents).hexdigest()
    if actual_sha256 != memory_exception["evidence_sha256"]:
        raise ValueError("memory exception evidence SHA-256 does not match")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(contents.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("memory exception evidence must be readable strict JSON") from error
    if not isinstance(payload, Mapping) or payload.get("model") != "convnext_tiny":
        raise ValueError("memory exception evidence has the wrong model identity")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise ValueError("memory exception evidence must contain two failed attempts")
    return contents


def _copy_memory_exception_evidence(
    run_dir: Path, resolved: Mapping[str, Any], project_root: Path
) -> None:
    memory_exception = resolved.get("memory_exception")
    if not isinstance(memory_exception, Mapping):
        return
    contents = _verified_memory_evidence_bytes(project_root, memory_exception)
    destination = run_dir / MEMORY_EVIDENCE_BUNDLE_NAME
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(contents)
        temporary.replace(destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != memory_exception[
            "evidence_sha256"
        ]:
            raise ValueError("memory exception run evidence SHA-256 does not match")
    finally:
        temporary.unlink(missing_ok=True)


def _memory_exception_note_lines(resolved: Mapping[str, Any]) -> list[str]:
    memory_exception = resolved.get("memory_exception")
    if not isinstance(memory_exception, Mapping):
        return []
    return [
        "Memory exception: "
        f"microbatch_size={memory_exception['microbatch_size']}, "
        "gradient_accumulation_steps="
        f"{memory_exception['gradient_accumulation_steps']}, "
        f"effective_batch_size={memory_exception['effective_batch_size']}",
        f"Memory exception reason: {memory_exception['reason']}",
        "Memory exception evidence: "
        f"{memory_exception['evidence_artifact']} "
        f"sha256={memory_exception['evidence_sha256']}",
        f"Memory exception run copy: {MEMORY_EVIDENCE_BUNDLE_NAME}",
    ]


def _validate_data_config(data: Mapping[str, Any]) -> None:
    missing = {"project_root", "dataset_root", "schema_path"} - set(data)
    if missing:
        raise ValueError(f"data config is missing required keys: {sorted(missing)}")
    unknown = set(data) - DATA_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown data config keys: {sorted(unknown)}")
    for key in ("project_root", "dataset_root", "schema_path"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} must be a non-empty path string")
    for key in ("workspace_root", "artifacts_root", "logs_root"):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            raise ValueError(f"{key} must be a non-empty path string")
    for key in ("preprocessing", "augmentation", "loader", "audit", "provenance", "split_policy"):
        if key in data and not isinstance(data[key], Mapping):
            raise ValueError(f"{key} must be a mapping")
    for key, allowed in NESTED_DATA_KEYS.items():
        unknown_nested = set(data.get(key, {})) - allowed
        if unknown_nested:
            raise ValueError(f"unknown data config {key} keys: {sorted(unknown_nested)}")
    for key in ("source_read_only", "labels_verified"):
        if key in data and type(data[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("image_size", "seed"):
        if key in data and (type(data[key]) is not int or int(data[key]) <= 0):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("official_splits", "image_extensions"):
        if key in data and (
            not isinstance(data[key], list)
            or not data[key]
            or any(not isinstance(value, str) or not value for value in data[key])
        ):
            raise ValueError(f"{key} must be a non-empty list of strings")

    preprocessing = data.get("preprocessing", {})
    for key in (
        "preserve_aspect_ratio",
        "pad_to_square",
        "percentile_clipping",
        "force_grayscale",
    ):
        if key in preprocessing and type(preprocessing[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if "channels" in preprocessing:
        _positive_int(preprocessing, "channels")
    if "pad_value" in preprocessing:
        _finite_number(preprocessing, "pad_value", positive=False)
    if "normalize" in preprocessing and (
        not isinstance(preprocessing["normalize"], str) or not preprocessing["normalize"]
    ):
        raise ValueError("normalize must be a non-empty string")

    augmentation = data.get("augmentation", {})
    for key in ("horizontal_flip", "vertical_flip", "clahe"):
        if key in augmentation and type(augmentation[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in (
        "rotation_limit",
        "translate_limit",
        "scale_limit",
        "brightness_contrast_limit",
    ):
        if key in augmentation:
            _finite_number(augmentation, key, positive=False)
    for key in ("gamma_limit", "gaussian_noise_std_range"):
        value = augmentation.get(key)
        if value is not None and (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            raise ValueError(f"{key} must be a two-number list")

    loader = data.get("loader", {})
    if "batch_size" in loader:
        _positive_int(loader, "batch_size")
    if "num_workers" in loader and (
        type(loader["num_workers"]) is not int or loader["num_workers"] < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    for key in ("pin_memory", "persistent_workers"):
        if key in loader and type(loader[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if "sampling" in loader and (
        not isinstance(loader["sampling"], str) or not loader["sampling"]
    ):
        raise ValueError("sampling must be a non-empty string")


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _validate_schema_semantics(schema: Mapping[str, Any]) -> None:
    binary = schema.get("binary")
    if not isinstance(binary, Mapping) or not (
        binary.get("enabled") is True
        and binary.get("semantics_verified") is True
        and _strict_equal(binary.get("classes"), {0: "normal", 1: "bone_tumour"})
    ):
        raise ValueError("label schema semantics do not match the verified binary task")
    for section in ("subtype", "risk", "concepts"):
        value = schema.get(section)
        if not isinstance(value, Mapping) or value.get("enabled") is not False:
            raise ValueError("label schema semantics enable unapproved optional supervision")
    localization = schema.get("localization")
    if not isinstance(localization, Mapping) or not (
        localization.get("boxes_verified") is False
        and localization.get("masks_verified") is False
    ):
        raise ValueError("label schema semantics enable unapproved spatial supervision")


def _load_verified_schema(path: Path) -> tuple[dict[str, Any], str]:
    contents = path.read_bytes()
    payload = yaml.safe_load(contents)
    if not isinstance(payload, dict):
        raise ValueError("label schema must contain a YAML mapping")
    _validate_schema_semantics(payload)
    digest = hashlib.sha256(contents).hexdigest()
    if digest != EXPECTED_SCHEMA_SHA256:
        raise ValueError("label schema identity does not match the verified Phase 4 schema")
    return payload, digest


def is_final_run_config(config: Mapping[str, Any]) -> bool:
    """Return whether a sanitized run record matches the exact Phase 5 policy."""
    data = config.get("data")
    expected_policy = dict(FINAL_POLICY)
    expected_loader_policy = dict(LOADER_POLICY)
    expected_exception = (
        CONVNEXT_MEMORY_EXCEPTION if config.get("model") == "convnext_tiny" else None
    )
    if expected_exception is not None:
        expected_policy["batch_size"] = expected_exception["microbatch_size"]
        expected_policy["gradient_accumulation_steps"] = expected_exception[
            "gradient_accumulation_steps"
        ]
        expected_loader_policy["batch_size"] = expected_exception["microbatch_size"]
    return bool(
        config.get("smoke_overrides") == {}
        and all(_strict_equal(config.get(key), value) for key, value in expected_policy.items())
        and (
            _strict_equal(config.get("memory_exception"), expected_exception)
            if expected_exception is not None
            else "memory_exception" not in config
        )
        and config.get("schema_sha256") == EXPECTED_SCHEMA_SHA256
        and isinstance(data, Mapping)
        and _strict_equal(data.get("preprocessing"), PREPROCESSING_POLICY)
        and _strict_equal(data.get("augmentation"), AUGMENTATION_POLICY)
        and _strict_equal(data.get("loader"), expected_loader_policy)
    )


def _is_sanitized_run_record(config: Mapping[str, Any], alias: str) -> bool:
    expected_keys = RECORD_CONFIG_KEYS | (
        {"memory_exception"} if alias == "convnext_tiny" else set()
    )
    if set(config) != expected_keys:
        return False
    expected_paths = {
        "project_root": ".",
        "model_config": f"configs/baseline_{alias}.yaml",
        "common_config": "configs/train.yaml",
        "data_config": "configs/data.yaml",
        "output_root": "experiments/baselines",
        "dataset_root": "dataset/Main",
        "split_dir": "workspace/splits",
        "schema_path": "workspace/annotations/label_schema.yaml",
    }
    if any(config.get(key) != value for key, value in expected_paths.items()):
        return False
    data = config.get("data")
    if not isinstance(data, Mapping) or set(data) != {
        "dataset_root",
        "image_size",
        "seed",
        "preprocessing",
        "augmentation",
        "loader",
    }:
        return False
    if not _strict_equal(data.get("dataset_root"), config.get("dataset_root")):
        return False
    if not _strict_equal(data.get("image_size"), config.get("image_size")) or not _strict_equal(
        data.get("seed"), config.get("seed")
    ):
        return False
    for key, allowed in NESTED_DATA_KEYS.items():
        value = data.get(key)
        if not isinstance(value, Mapping) or set(value) - allowed:
            return False
    loader = data["loader"]
    if any(
        not _strict_equal(loader.get(key), config.get(key))
        for key in ("batch_size", "num_workers", "pin_memory", "persistent_workers")
    ):
        return False
    return True


def resolve_config(
    model_config_path: Path,
    *,
    epochs_override: int | None = None,
    pretrained_override: bool | None = None,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    """Resolve and validate a model/common/data configuration chain."""

    model_path = Path(model_config_path).resolve()
    if model_path.parent.name.casefold() != "configs":
        raise ValueError("model config must be inside the project configs directory")
    inferred_project_root = model_path.parent.parent.resolve()
    model_config = _load_yaml(model_path, name="model config")
    model = model_config.get("model")
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    if model not in APPROVED_MODELS:
        approved = ", ".join(APPROVED_MODELS)
        raise ValueError(f"Unknown baseline {model!r}; choose one of: {approved}")
    expected_model_keys = {"model", "common_config"} | (
        {"memory_exception"} if model == "convnext_tiny" else set()
    )
    if set(model_config) != expected_model_keys:
        raise ValueError(
            f"model config keys must be exactly: {sorted(expected_model_keys)}"
        )
    memory_exception = model_config.get("memory_exception")
    if model == "convnext_tiny" and not _strict_equal(
        memory_exception, CONVNEXT_MEMORY_EXCEPTION
    ):
        raise ValueError("convnext_tiny memory_exception does not match the approved evidence")
    common_path = _resolve_declared_path(model_config.get("common_config"), model_path)
    common = _load_yaml(common_path, name="common config")
    unknown_common_keys = set(common) - COMMON_CONFIG_KEYS
    if unknown_common_keys:
        raise ValueError(f"unknown common config keys: {sorted(unknown_common_keys)}")
    missing_common_keys = COMMON_CONFIG_KEYS - set(common)
    if missing_common_keys:
        raise ValueError(f"common config is missing keys: {sorted(missing_common_keys)}")
    for key in ("data_config", "output_root"):
        if not isinstance(common[key], str) or not common[key].strip():
            raise ValueError(f"{key} must be a non-empty path string")

    data_path = _resolve_declared_path(common["data_config"], common_path)
    data_config = _load_yaml(data_path, name="data config")
    _validate_data_config(data_config)
    project_root = _resolve_declared_path(data_config["project_root"], data_path)
    if project_root != inferred_project_root:
        raise ValueError("project_root must match the model config project boundary")
    _require_within(model_path, project_root, name="model config")
    _require_within(common_path, project_root, name="common config")
    _require_within(data_path, project_root, name="data config")

    output_root = (project_root / str(common["output_root"])).resolve()
    _require_within(output_root, project_root, name="output_root")
    dataset_root = (project_root / str(data_config.get("dataset_root", ""))).resolve()
    _require_within(dataset_root, project_root, name="dataset_root")
    source_root = (project_root / "dataset" / "Main").resolve()
    if dataset_root != source_root:
        raise ValueError("dataset_root must be the project's immutable dataset/Main")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset_root: {dataset_root}")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("output_root must not be inside dataset/Main")
    expected_output_root = (project_root / "experiments" / "baselines").resolve()
    if output_root != expected_output_root:
        raise ValueError("output_root must be experiments/baselines")
    workspace_value = data_config.get("workspace_root", "workspace")
    workspace_root = (project_root / str(workspace_value)).resolve()
    _require_within(workspace_root, project_root, name="workspace_root")
    expected_workspace = (project_root / "workspace").resolve()
    if workspace_root != expected_workspace:
        raise ValueError("workspace_root must be the project workspace directory")
    schema_path = (project_root / str(data_config["schema_path"])).resolve()
    _require_within(schema_path, project_root, name="schema_path")
    expected_schema_path = (workspace_root / "annotations" / "label_schema.yaml").resolve()
    if schema_path != expected_schema_path:
        raise ValueError("schema_path must be the canonical label_schema.yaml in workspace")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    schema, schema_sha256 = _load_verified_schema(schema_path)
    split_dir = (workspace_root / "splits").resolve()
    _require_within(split_dir, project_root, name="split_dir")
    for split in SPLIT_NAMES:
        split_path = split_dir / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {split_path}")

    if common["device"] != "cuda":
        raise ValueError("device must be cuda for Phase 5 baselines")
    if common["selection_split"] != "val":
        raise ValueError("selection_split must be val")
    if common["selection_metric"] != "AUROC":
        raise ValueError("selection_metric must be AUROC")
    if common["optimizer"] != "adamw" or common["scheduler"] != "cosine":
        raise ValueError("optimizer and scheduler must be adamw and cosine")
    if common["allow_cpu_training"] is not False:
        raise ValueError("allow_cpu_training must be false")
    for key in ("amp", "pin_memory", "persistent_workers"):
        if type(common[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if common["amp"] is not True:
        raise ValueError("amp must be true")
    for key in (
        "image_size",
        "batch_size",
        "epochs",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "seed",
    ):
        _positive_int(common, key)
    for key in ("learning_rate", "gradient_clip_norm"):
        _finite_number(common, key, positive=True)
    _finite_number(common, "weight_decay", positive=False)
    if type(common["warmup_epochs"]) is not int or common["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs must be a non-negative integer")
    if type(common["num_workers"]) is not int or common["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if common["persistent_workers"] and common["num_workers"] == 0:
        raise ValueError("persistent_workers requires positive num_workers")
    if epochs_override is not None and (type(epochs_override) is not int or epochs_override <= 0):
        raise ValueError("epochs override must be a positive integer")
    if pretrained_override is not None and type(pretrained_override) is not bool:
        raise ValueError("pretrained override must be boolean")
    if limit_batches is not None and (type(limit_batches) is not int or limit_batches <= 0):
        raise ValueError("limit_batches must be a positive integer")

    record = dict(common)
    record["model"] = model
    record["project_root"] = "."
    record["model_config"] = _relative_path(model_path, project_root)
    record["common_config"] = _relative_path(common_path, project_root)
    record["data_config"] = _relative_path(data_path, project_root)
    record["output_root"] = _relative_path(output_root, project_root)
    record["dataset_root"] = _relative_path(dataset_root, project_root)
    record["split_dir"] = _relative_path(split_dir, project_root)
    record["schema_path"] = _relative_path(schema_path, project_root)
    record["schema_sha256"] = schema_sha256
    record["pretrained"] = True if pretrained_override is None else pretrained_override
    record["limit_batches"] = limit_batches
    if memory_exception is not None:
        record["batch_size"] = memory_exception["microbatch_size"]
        record["gradient_accumulation_steps"] = memory_exception[
            "gradient_accumulation_steps"
        ]
        record["memory_exception"] = json.loads(json.dumps(memory_exception))
    smoke_overrides: dict[str, object] = {}
    if epochs_override is not None:
        record["epochs"] = epochs_override
        record["warmup_epochs"] = min(int(record["warmup_epochs"]), epochs_override)
        smoke_overrides["epochs"] = epochs_override
    if pretrained_override is not None:
        smoke_overrides["pretrained"] = pretrained_override
    if limit_batches is not None:
        smoke_overrides["limit_batches"] = limit_batches
    record["smoke_overrides"] = smoke_overrides

    loader = dict(data_config.get("loader", {}))
    loader.update(
        {
            "batch_size": record["batch_size"],
            "num_workers": record["num_workers"],
            "pin_memory": record["pin_memory"],
            "persistent_workers": record["persistent_workers"],
        }
    )
    runtime_data = {
        "dataset_root": str(dataset_root),
        "image_size": record["image_size"],
        "seed": record["seed"],
        "preprocessing": dict(data_config.get("preprocessing", {})),
        "augmentation": dict(data_config.get("augmentation", {})),
        "loader": loader,
    }
    record["data"] = {
        **runtime_data,
        "dataset_root": _relative_path(dataset_root, project_root),
    }
    record["run_type"] = "final" if is_final_run_config(record) else "smoke"
    runtime = {
        "project_root": str(project_root),
        "model_config_path": str(model_path),
        "common_config_path": str(common_path),
        "data_config_path": str(data_path),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "schema_path": str(schema_path),
        "schema": schema,
        "data": runtime_data,
    }
    return {"record": record, "runtime": runtime}


class _LimitedLoader:
    def __init__(self, loader: Iterable[Any], limit: int) -> None:
        self._loader = loader
        self.limit = limit

    def __iter__(self) -> Iterator[Any]:
        return itertools.islice(iter(self._loader), self.limit)

    def __getattr__(self, name: str) -> object:
        return getattr(self._loader, name)


def _limit_loader(loader: Iterable[Any], limit: int | None) -> Iterable[Any]:
    return loader if limit is None else _LimitedLoader(loader, limit)


def _new_run_directory(output_root: Path, model: str) -> Path:
    parent = output_root / model
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = parent / f"{stamp}-{uuid4().hex[:8]}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a unique baseline run directory")


def _write_environment(path: Path) -> None:
    import torch

    packages = {}
    for name in ("torch", "timm", "pandas", "scikit-learn", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    lines = [
        f"python={platform.python_version()}",
        f"platform={platform.system()} {platform.release()}",
        *(f"{name}={value}" for name, value in packages.items()),
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda or 'unavailable'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_git_state(path: Path, project_root: Path) -> None:
    try:
        sha = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        contents = f"commit={sha}\ndirty={str(dirty).lower()}\n"
    except (OSError, subprocess.CalledProcessError):
        contents = "commit=unavailable\ndirty=unavailable\n"
    path.write_text(contents, encoding="utf-8", newline="\n")


def _write_notes(path: Path, resolved: Mapping[str, Any], *, status: str, detail: str) -> None:
    overrides = resolved["smoke_overrides"]
    run_kind = resolved["run_type"]
    memory_lines = _memory_exception_note_lines(resolved)
    path.write_text(
        "\n".join(
            [
                f"Status: {status}",
                f"Model: {resolved['model']}",
                f"Run type: {run_kind}",
                f"Smoke overrides: {json.dumps(overrides, sort_keys=True)}",
                "Selection uses validation AUROC at a fixed probability threshold of 0.5.",
                "The test split is untouched during fitting and test is evaluated once "
                "after loading best.pt.",
                *memory_lines,
                detail,
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _prediction_frame(prediction: Mapping[str, Any]) -> Any:
    import pandas as pd
    import torch

    logits = torch.as_tensor(prediction["logits"]).detach().cpu().to(torch.float64)
    targets = torch.as_tensor(prediction["targets"]).detach().cpu().to(torch.long)
    sample_ids = [str(value) for value in prediction["sample_ids"]]
    if logits.ndim != 1 or targets.shape != logits.shape or len(sample_ids) != len(logits):
        raise ValueError("prediction payload has inconsistent lengths")
    probabilities = torch.sigmoid(logits)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target": targets.tolist(),
            "logit": logits.tolist(),
            "probability": probabilities.tolist(),
            "prediction": (probabilities >= 0.5).to(torch.long).tolist(),
        }
    )


def _set_reproducibility(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _remove_result_files(run_dir: Path) -> None:
    for name in (
        "metrics.json",
        "history.csv",
        "confusion_matrix.csv",
        "test_predictions.csv",
        "best_checkpoint_path.txt",
    ):
        run_dir.joinpath(name).unlink(missing_ok=True)


def run_baseline(
    config_path: Path,
    *,
    epochs_override: int | None = None,
    pretrained_override: bool | None = None,
    limit_batches: int | None = None,
    dependencies: RunDependencies | None = None,
) -> Path:
    """Execute one immutable train/validate/test baseline run."""

    resolved = resolve_config(
        config_path,
        epochs_override=epochs_override,
        pretrained_override=pretrained_override,
        limit_batches=limit_batches,
    )
    record = resolved["record"]
    runtime = resolved["runtime"]
    dependencies = dependencies or _default_dependencies()
    if not dependencies.cuda_available():
        raise RuntimeError("CUDA is required for Phase 5 baseline training and is unavailable")

    import pandas as pd
    import torch

    from bone_cancer_ai.evaluation.reports import (
        atomic_write_csv,
        atomic_write_json,
        confusion_matrix_frame,
        update_baseline_summary,
    )

    project_root = Path(runtime["project_root"])
    run_dir = _new_run_directory(Path(runtime["output_root"]), str(record["model"]))
    notes_path = run_dir / "notes.md"

    try:
        _copy_memory_exception_evidence(run_dir, record, project_root)
        run_dir.joinpath("config.yaml").write_text(
            yaml.safe_dump(record, sort_keys=False), encoding="utf-8", newline="\n"
        )
        _write_environment(run_dir / "environment.txt")
        _write_git_state(run_dir / "git_commit.txt", project_root)
        _write_notes(notes_path, record, status="Running", detail="Run initialized.")
        _set_reproducibility(int(record["seed"]))
        data_module = dependencies.data_module_type(
            Path(runtime["split_dir"]), runtime["data"], runtime["schema"]
        )
        data_module.setup()
        train_loader = _limit_loader(
            data_module.loader("fit_train"), record["limit_batches"]
        )
        val_loader = _limit_loader(data_module.loader("val"), record["limit_batches"])
        test_loader = _limit_loader(data_module.loader("test"), record["limit_batches"])
        model = dependencies.create_model(
            name=record["model"],
            pretrained=record["pretrained"],
            image_size=record["image_size"],
        )
        parameter_count = int(dependencies.count_parameters(model))
        pos_weight = dependencies.class_weight(data_module.frames["fit_train"])
        trainer = dependencies.trainer_type(
            model,
            model_alias=record["model"],
            resolved_config=record,
            checkpoint_dir=run_dir,
            pos_weight=pos_weight,
            device=record["device"],
        )
        dependencies.cuda_reset_peak()
        fit_result = trainer.fit(train_loader, val_loader)
        best_path = run_dir / "best.pt"
        dependencies.load_checkpoint(
            best_path,
            model=model,
            expected_model_alias=record["model"],
            map_location=record["device"],
            expected_resolved_config=record,
            expected_selection_metric="AUROC",
        )
        validation_prediction = trainer.predict(val_loader)
        validation_metrics = dependencies.metrics(
            validation_prediction["logits"], validation_prediction["targets"], threshold=0.5
        )
        test_prediction = trainer.predict(test_loader)
        test_metrics = dependencies.metrics(
            test_prediction["logits"], test_prediction["targets"], threshold=0.5
        )
        latency = dependencies.latency(
            model,
            torch.zeros(1, 3, int(record["image_size"]), int(record["image_size"])),
            record["device"],
        )
        resources = {
            "trainable_parameters": parameter_count,
            "best_checkpoint_bytes": best_path.stat().st_size,
            "cuda_peak_memory_bytes": int(dependencies.cuda_peak_memory()),
            "latency": latency,
        }
        metrics_payload = {
            "metadata": {
                "model": record["model"],
                "seed": record["seed"],
                "run_type": record["run_type"],
                "best_epoch": int(fit_result["best_epoch"]),
            },
            "selection": {"split": "val", "metric": "AUROC", "threshold": 0.5},
            "validation": validation_metrics,
            "test": test_metrics,
            "prediction_counts": {
                "validation_valid": int(validation_prediction["valid_count"]),
                "validation_rejected": int(validation_prediction["rejected_count"]),
                "test_valid": int(test_prediction["valid_count"]),
                "test_rejected": int(test_prediction["rejected_count"]),
            },
            "resources": resources,
        }
        publication_note_lines = notes_path.read_text(encoding="utf-8").splitlines()
        _validate_memory_exception_bundle(
            run_dir, record, project_root, publication_note_lines
        )
        atomic_write_csv(pd.DataFrame(fit_result["history"]), run_dir / "history.csv")
        atomic_write_csv(
            confusion_matrix_frame(test_metrics), run_dir / "confusion_matrix.csv"
        )
        atomic_write_csv(_prediction_frame(test_prediction), run_dir / "test_predictions.csv")
        atomic_write_json(metrics_payload, run_dir / "metrics.json")
        run_dir.joinpath("best_checkpoint_path.txt").write_text(
            "best.pt\n", encoding="utf-8", newline="\n"
        )
        _write_notes(notes_path, record, status="Complete", detail="Artifacts are complete.")
        publication_note_lines = notes_path.read_text(encoding="utf-8").splitlines()
        _validate_memory_exception_bundle(
            run_dir, record, project_root, publication_note_lines
        )
        summary_row = {
            "model": record["model"],
            "seed": record["seed"],
            "best_epoch": int(fit_result["best_epoch"]),
            **{column: test_metrics[column] for column in BASELINE_SUMMARY_COLUMNS[3:10]},
            "params": parameter_count,
            "latency_ms": latency["latency_ms"],
        }
        if is_final_run_config(record) and _is_sanitized_run_record(
            record, str(record["model"])
        ):
            update_baseline_summary(
                project_root / "artifacts" / "metrics" / "baseline_summary.csv", summary_row
            )
        return run_dir
    except torch.cuda.OutOfMemoryError as error:
        guidance = (
            "CUDA out of memory. Reduce per-device batch size and increase gradient accumulation "
            "to preserve effective batch size before considering a resolution change."
        )
        _remove_result_files(run_dir)
        _write_notes(notes_path, record, status="Failed", detail=guidance)
        raise torch.cuda.OutOfMemoryError(f"{error}. {guidance}") from error
    except Exception as error:
        _remove_result_files(run_dir)
        detail = f"{type(error).__name__}: {str(error).splitlines()[0][:500]}"
        _write_notes(notes_path, record, status="Failed", detail=detail)
        raise


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _validate_memory_exception_bundle(
    run_dir: Path,
    config: Mapping[str, Any],
    project_root: Path,
    note_lines: list[str],
) -> None:
    memory_exception = config.get("memory_exception")
    expected_note_lines = _memory_exception_note_lines(config)
    actual_note_lines = [
        line for line in note_lines if line.startswith(MEMORY_EXCEPTION_NOTE_PREFIXES)
    ]
    if actual_note_lines != expected_note_lines:
        raise ValueError("memory exception notes are missing, reordered, or tampered")
    if memory_exception is None:
        return
    if not isinstance(memory_exception, Mapping):
        raise ValueError("memory exception record is malformed")
    if config.get("model") != "convnext_tiny" or not _strict_equal(
        memory_exception, CONVNEXT_MEMORY_EXCEPTION
    ):
        raise ValueError("memory exception record does not match the approved evidence")
    data = config.get("data")
    if not (
        config.get("batch_size") == memory_exception["microbatch_size"]
        and config.get("gradient_accumulation_steps")
        == memory_exception["gradient_accumulation_steps"]
        and isinstance(data, Mapping)
        and isinstance(data.get("loader"), Mapping)
        and data["loader"].get("batch_size") == memory_exception["microbatch_size"]
    ):
        raise ValueError("memory exception record has inconsistent batch settings")
    source_contents = _verified_memory_evidence_bytes(project_root, memory_exception)
    copy_path = run_dir / MEMORY_EVIDENCE_BUNDLE_NAME
    try:
        copy_contents = copy_path.read_bytes()
        copy_payload = _strict_json(copy_path)
    except (OSError, ValueError) as error:
        raise ValueError("memory exception evidence copy is missing or unreadable") from error
    if copy_contents != source_contents or not isinstance(copy_payload, Mapping):
        raise ValueError("memory exception evidence copy does not match its verified source")


def _complete_final_candidate(
    run_dir: Path, alias: str, seed: int, project_root: Path
) -> bool:
    present = {path.name for path in run_dir.iterdir() if path.is_file()}
    if not REQUIRED_RUN_FILES <= present:
        return False
    try:
        config = _load_yaml(run_dir / "config.yaml", name="resolved run config")
        environment = run_dir.joinpath("environment.txt").read_text(encoding="utf-8").strip()
        git_state = run_dir.joinpath("git_commit.txt").read_text(encoding="utf-8").strip()
        note_lines = run_dir.joinpath("notes.md").read_text(encoding="utf-8").splitlines()
        _validate_memory_exception_bundle(run_dir, config, project_root, note_lines)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(
        config.get("model") == alias
        and config.get("seed") == seed
        and config.get("run_type") == "final"
        and _is_sanitized_run_record(config, alias)
        and is_final_run_config(config)
        and environment
        and git_state
        and [line for line in note_lines if line.startswith("Status:")] == ["Status: Complete"]
    )


def _validate_run_bundle(run_dir: Path, alias: str, summary_row: Any) -> None:
    import pandas as pd

    present = {path.name for path in run_dir.iterdir() if path.is_file()}
    missing = REQUIRED_RUN_FILES - present
    if missing:
        raise ValueError(f"run {run_dir.name} is missing artifacts: {sorted(missing)}")
    config = _load_yaml(run_dir / "config.yaml", name="resolved run config")
    if config.get("model") != alias or config.get("seed") != int(summary_row["seed"]):
        raise ValueError(f"run {run_dir.name} has the wrong model or seed")
    if (
        config.get("run_type") != "final"
        or not _is_sanitized_run_record(config, alias)
        or not is_final_run_config(config)
    ):
        raise ValueError(f"run {run_dir.name} violates the exact final policy")
    note_lines = run_dir.joinpath("notes.md").read_text(encoding="utf-8").splitlines()
    project_root = run_dir.parents[3]
    _validate_memory_exception_bundle(run_dir, config, project_root, note_lines)
    for evidence_name in ("environment.txt", "git_commit.txt"):
        if not run_dir.joinpath(evidence_name).read_text(encoding="utf-8").strip():
            raise ValueError(f"run {run_dir.name} has empty {evidence_name} evidence")
    notes = run_dir.joinpath("notes.md").read_text(encoding="utf-8").splitlines()
    if [line for line in notes if line.startswith("Status:")] != ["Status: Complete"]:
        raise ValueError(f"run {run_dir.name} lacks exact Complete Status evidence")
    metrics = _strict_json(run_dir / "metrics.json")
    if not isinstance(metrics, dict) or metrics.get("selection") != {
        "split": "val",
        "metric": "AUROC",
        "threshold": 0.5,
    }:
        raise ValueError(f"run {run_dir.name} has invalid selection evidence")
    for section in ("validation", "test", "resources"):
        if not isinstance(metrics.get(section), dict):
            raise ValueError(f"run {run_dir.name} has invalid metrics JSON")
    if metrics.get("metadata") != {
        "model": alias,
        "seed": int(summary_row["seed"]),
        "run_type": "final",
        "best_epoch": int(summary_row["best_epoch"]),
    }:
        raise ValueError(f"run {run_dir.name} metrics metadata does not match the summary")
    history = pd.read_csv(run_dir / "history.csv")
    confusion = pd.read_csv(run_dir / "confusion_matrix.csv")
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    if history.empty or confusion.shape != (2, 2) or predictions.empty:
        raise ValueError(f"run {run_dir.name} contains empty report artifacts")
    if predictions.columns.tolist() != [
        "sample_id",
        "target",
        "logit",
        "probability",
        "prediction",
    ]:
        raise ValueError(f"run {run_dir.name} has invalid test predictions")
    checkpoint_name = run_dir.joinpath("best_checkpoint_path.txt").read_text(
        encoding="utf-8"
    ).strip()
    checkpoint = (run_dir / checkpoint_name).resolve()
    if checkpoint != (run_dir / "best.pt").resolve() or not checkpoint.is_file():
        raise ValueError(f"run {run_dir.name} has an invalid best checkpoint target")
    if (
        run_dir.joinpath("best.pt").stat().st_size == 0
        or run_dir.joinpath("last.pt").stat().st_size == 0
    ):
        raise ValueError(f"run {run_dir.name} has an empty checkpoint")
    test_metrics = metrics["test"]
    resources = metrics["resources"]
    expected = {
        "best_epoch": int(history["epoch"].loc[history["val_AUROC"].idxmax()]),
        **{column: test_metrics[column] for column in BASELINE_SUMMARY_COLUMNS[3:10]},
        "params": resources["trainable_parameters"],
        "latency_ms": resources["latency"]["latency_ms"],
    }
    for column, value in expected.items():
        actual = summary_row[column]
        if value is None:
            if not pd.isna(actual):
                raise ValueError(f"summary {column} does not match run {run_dir.name}")
        elif not math.isclose(float(actual), float(value), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"summary {column} does not match run {run_dir.name}")


def validate_phase5_artifacts(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Validate four final baseline summaries and their latest complete bundles."""

    root = Path(project_root).resolve()
    summary_path = root / "artifacts" / "metrics" / "baseline_summary.csv"
    if not summary_path.is_file():
        raise ValueError("Phase 5 requires four approved model/seed summary rows")
    import pandas as pd

    summary = pd.read_csv(summary_path)
    if summary.columns.tolist() != BASELINE_SUMMARY_COLUMNS:
        raise ValueError("baseline summary schema is invalid")
    if len(summary) != 4 or set(summary["model"]) != set(APPROVED_MODELS):
        raise ValueError("Phase 5 requires four approved model/seed summary rows")
    if summary[["model", "seed"]].duplicated().any() or not summary["seed"].eq(42).all():
        raise ValueError("Phase 5 requires four approved model/seed summary rows")
    for _, row in summary.iterrows():
        alias = str(row["model"])
        model_root = root / "experiments" / "baselines" / alias
        candidates = (
            sorted((path for path in model_root.iterdir() if path.is_dir()), reverse=True)
            if model_root.is_dir()
            else []
        )
        complete_final_runs = [
            candidate
            for candidate in candidates
            if _complete_final_candidate(candidate, alias, int(row["seed"]), root)
        ]
        if not complete_final_runs:
            raise ValueError(f"no complete final run exists for {alias}")
        _validate_run_bundle(complete_final_runs[0], alias, row)
    return {"models": 4, "runs": 4, "status": "valid"}


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrained", type=_parse_bool)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--validate-artifacts", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_artifacts:
        if args.config is not None or any(
            value is not None for value in (args.epochs, args.pretrained, args.limit_batches)
        ):
            parser.error("--validate-artifacts cannot be combined with training arguments")
        print(json.dumps(validate_phase5_artifacts(PROJECT_ROOT), indent=2))
        return 0
    if args.config is None:
        parser.error("Phase 5 baseline training requires --config")
    run_dir = run_baseline(
        args.config,
        epochs_override=args.epochs,
        pretrained_override=args.pretrained,
        limit_batches=args.limit_batches,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
