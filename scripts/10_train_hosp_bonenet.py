"""Train binary-only HOSP-BoneNet in frozen Stage A and Stage D runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_GENERATION_ID = (
    "de64a489fff7202ec39d442e4b26a5a5c0d7e2c7db8ed7a8587601ef1cd28590"
)
EXPECTED_INPUT_SHA256 = {
    "fit_train": "e30c120664d94e2342e1ad93ae9c072222467371ffb7743f059cd215499e4830",
    "val": "1eb4f89a9aeb50cba5b298bd003ea5f599664f287f173051dbd6dbcc14c77973",
    "schema": "6248ff8274f5ad5d26d046f851d672f33390d0a8ef07d6c4b6a0eee11dcefe95",
}
MODEL_CONFIG_KEYS = {
    "status",
    "train_config",
    "data_config",
    "model",
    "loss_weights",
    "verified_supervision",
}
TRAIN_CONFIG_KEYS = {
    "output_root",
    "image_size",
    "microbatch_size",
    "effective_batch_size",
    "gradient_accumulation_steps",
    "gradient_clip_norm",
    "weight_decay",
    "scheduler",
    "amp",
    "seed",
    "device",
    "allow_cpu_training",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "selection_split",
    "selection_metric",
    "stages",
}
MODEL_KEYS = {
    "name",
    "backbone",
    "pretrained",
    "feature_dim",
    "pyramid_channels",
    "dropout",
    "pooling",
    "localization_enabled",
    "subtype_enabled",
    "num_subtypes",
    "risk_enabled",
    "num_risk_classes",
    "concepts_enabled",
    "num_concepts",
}
LOSS_KEYS = {"binary", "subtype", "risk", "segmentation", "concepts", "grounding", "hierarchy"}
STAGE_A_POLICY = {
    "enabled": True,
    "name": "A_binary_global_local",
    "epochs": 40,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "warmup_epochs": 2,
    "early_stopping_patience": 8,
}
STAGE_D_POLICY = {
    "enabled": True,
    "name": "D_low_lr_finetune",
    "epochs": 10,
    "optimizer": "adamw",
    "learning_rate": 0.00001,
    "warmup_epochs": 0,
    "early_stopping_patience": 4,
    "initialize_from": "A.best",
}
STAGE_REQUIRED_FILES = {
    "best.pt",
    "last.pt",
    "history.csv",
    "training_curves.png",
    "validation_predictions.csv",
}
ROOT_REQUIRED_FILES = {
    "config.yaml",
    "environment.txt",
    "git_commit.txt",
    "notes.md",
    "status.json",
    "selection.json",
    "metrics.json",
    "best_checkpoint_path.txt",
}
SELECTED_CHECKPOINT_ALIAS = "hosp_bonenet.phase7.selected"


class RunDependencies:
    """Inject slow framework boundaries while keeping orchestration real in tests."""

    def __init__(
        self,
        *,
        data_module_type: type,
        trainer_type: type,
        create_model: Callable[..., Any],
        class_weight: Callable[[Any], Any],
        load_checkpoint: Callable[..., Mapping[str, Any]],
        metrics: Callable[..., dict[str, object]],
        cuda_available: Callable[[], bool],
        plot_history: Callable[[list[dict[str, Any]], Path], None],
    ) -> None:
        self.data_module_type = data_module_type
        self.trainer_type = trainer_type
        self.create_model = create_model
        self.class_weight = class_weight
        self.load_checkpoint = load_checkpoint
        self.metrics = metrics
        self.cuda_available = cuda_available
        self.plot_history = plot_history


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_exact_keys(values: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(f"{name} keys are invalid; missing={missing}, extra={extra}")


def validate_binary_model_policy(document: Mapping[str, Any]) -> None:
    """Reject every HOSP configuration outside released binary supervision."""

    if not isinstance(document, Mapping):
        raise ValueError("model config must be a mapping")
    _require_exact_keys(document, MODEL_CONFIG_KEYS, "model config")
    if document["status"] != "phase_6_verified":
        raise ValueError("HOSP model status must be phase_6_verified")
    if document["verified_supervision"] != ["binary"]:
        raise ValueError("verified supervision must contain binary only")
    model = document["model"]
    losses = document["loss_weights"]
    if not isinstance(model, Mapping) or not isinstance(losses, Mapping):
        raise ValueError("model and loss_weights must be mappings")
    _require_exact_keys(model, MODEL_KEYS, "model")
    _require_exact_keys(losses, LOSS_KEYS, "loss_weights")
    if model["name"] != "hosp_bonenet" or model["pretrained"] is not True:
        raise ValueError("production model must be pretrained hosp_bonenet")
    if model["localization_enabled"] is not True:
        raise ValueError("global/local fusion requires localization_enabled architecture")
    optional_switches = ("subtype_enabled", "risk_enabled", "concepts_enabled")
    optional_counts = ("num_subtypes", "num_risk_classes", "num_concepts")
    if any(model[key] is not False for key in optional_switches) or any(
        model[key] != 0 for key in optional_counts
    ):
        raise ValueError("all optional HOSP heads must remain disabled")
    if losses["binary"] != 1.0:
        raise ValueError("binary loss weight must be exactly 1.0")
    if any(losses[key] != 0.0 for key in LOSS_KEYS - {"binary"}):
        raise ValueError("every optional task loss must remain zero")


def validate_training_policy(document: Mapping[str, Any]) -> None:
    """Validate the frozen Phase 7 optimizer, staging, and isolation policy."""

    if not isinstance(document, Mapping):
        raise ValueError("training config must be a mapping")
    _require_exact_keys(document, TRAIN_CONFIG_KEYS, "training config")
    if document["device"] != "cuda":
        raise ValueError("device must be cuda")
    if document["allow_cpu_training"] is not False:
        raise ValueError("allow_cpu_training must be false")
    if document["amp"] is not True:
        raise ValueError("CUDA AMP must be enabled")
    if document["selection_split"] != "val":
        raise ValueError("selection_split must be val")
    if document["selection_metric"] != "AUROC":
        raise ValueError("selection_metric must be AUROC")
    if document["scheduler"] != "cosine":
        raise ValueError("scheduler must be cosine")
    if document["output_root"] != "experiments/hosp_bonenet":
        raise ValueError("output_root must be experiments/hosp_bonenet")
    integers = (
        "image_size",
        "microbatch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "seed",
    )
    for key in integers:
        if type(document[key]) is not int or document[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if document["image_size"] != 384 or document["seed"] != 42:
        raise ValueError("image_size and seed must remain 384 and 42")
    if (
        document["microbatch_size"] * document["gradient_accumulation_steps"]
        != document["effective_batch_size"]
        or document["effective_batch_size"] != 8
    ):
        raise ValueError("effective batch must equal microbatch times accumulation and be 8")
    if document["microbatch_size"] != 4 or document["gradient_accumulation_steps"] != 2:
        raise ValueError("approved microbatch policy is 4 with two accumulation steps")
    if not isinstance(document["gradient_clip_norm"], int | float) or not math.isclose(
        float(document["gradient_clip_norm"]), 1.0
    ):
        raise ValueError("gradient_clip_norm must be 1.0")
    if not isinstance(document["weight_decay"], int | float) or not math.isclose(
        float(document["weight_decay"]), 0.0001
    ):
        raise ValueError("weight_decay must be 0.0001")
    if type(document["num_workers"]) is not int or document["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    if document["pin_memory"] is not True or document["persistent_workers"] is not False:
        raise ValueError("loader memory policy is invalid")
    stages = document["stages"]
    if not isinstance(stages, Mapping) or set(stages) != {"A", "B", "C", "D"}:
        raise ValueError("stages must contain exactly A, B, C, and D")
    if not _strict_equal(stages["A"], STAGE_A_POLICY):
        raise ValueError("Stage A does not match the approved policy")
    if not _strict_equal(stages["D"], STAGE_D_POLICY):
        raise ValueError("Stage D does not match the approved policy")
    for key in ("B", "C"):
        stage = stages[key]
        if (
            not isinstance(stage, Mapping)
            or set(stage) != {"enabled", "reason"}
            or stage["enabled"] is not False
            or not isinstance(stage["reason"], str)
            or not stage["reason"].strip()
        ):
            raise ValueError(f"Stage {key} must be explicitly skipped with a reason")


def _sibling_config(reference: object, declaring_path: Path, name: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{name} reference must be a non-empty string")
    relative = Path(reference)
    if relative.is_absolute() or relative.name != reference or relative.suffix != ".yaml":
        raise ValueError(f"{name} must be a YAML file in the project configs directory")
    result = (declaring_path.parent / relative).resolve()
    if result.parent != declaring_path.parent.resolve():
        raise ValueError(f"{name} escapes the project configs directory")
    return result


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(split_dir: Path, schema_path: Path) -> dict[str, Any]:
    import pandas as pd

    manifest_hashes: dict[str, str] = {}
    generation_ids: set[str] = set()
    for split in ("fit_train", "val"):
        path = split_dir / f"{split}.csv"
        frame = pd.read_csv(path, usecols=["split_generation_id"])
        values = set(frame["split_generation_id"].dropna().astype(str))
        if len(values) != 1:
            raise ValueError(f"{split} manifest has invalid split_generation_id values")
        generation_ids.update(values)
        manifest_hashes[split] = _sha256(path)
    if len(generation_ids) != 1:
        raise ValueError("fit_train and val use different split generations")
    return {
        "split_generation_id": generation_ids.pop(),
        "manifest_sha256": manifest_hashes,
        "schema_sha256": _sha256(schema_path),
    }


def validate_recorded_inputs(
    record: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Require current manifests and schema to match the run's frozen input record."""

    expected = record.get("inputs")
    if not isinstance(expected, Mapping):
        raise ValueError("run config has no recorded input hashes")
    actual = _input_record(Path(runtime["split_dir"]), Path(runtime["schema_path"]))
    if not _strict_equal(actual, expected):
        raise ValueError("current input hash record does not match the frozen run inputs")
    return actual


def _validate_schema(schema: Mapping[str, Any]) -> None:
    binary = schema.get("binary")
    if not isinstance(binary, Mapping) or not (
        binary.get("enabled") is True and binary.get("semantics_verified") is True
    ):
        raise ValueError("verified binary schema is required")
    for key in ("subtype", "risk", "concepts"):
        section = schema.get(key)
        if not isinstance(section, Mapping) or section.get("enabled") is not False:
            raise ValueError("schema enables unreleased optional supervision")
    localization = schema.get("localization")
    if not isinstance(localization, Mapping) or not (
        localization.get("boxes_verified") is False and localization.get("masks_verified") is False
    ):
        raise ValueError("schema enables unreleased spatial supervision")


def resolve_config(model_config_path: Path) -> dict[str, Any]:
    """Resolve strict project-local config files into public record and runtime views."""

    from bone_cancer_ai.utils.config import load_config
    from bone_cancer_ai.utils.paths import ProjectPaths

    model_path = Path(model_config_path).resolve()
    if model_path.parent.name.casefold() != "configs":
        raise ValueError("model config must be inside a configs directory")
    project_root = model_path.parent.parent.resolve()
    model_document = load_config(model_path)
    validate_binary_model_policy(model_document)
    train_path = _sibling_config(model_document["train_config"], model_path, "train_config")
    data_path = _sibling_config(model_document["data_config"], model_path, "data_config")
    train_document = load_config(train_path)
    validate_training_policy(train_document)
    data_document = load_config(data_path)
    paths = ProjectPaths.from_config(data_document)
    if paths.project_root != project_root:
        raise ValueError("project_root must match the config project boundary")
    expected_dataset = (project_root / "dataset" / "Main").resolve()
    if paths.dataset_root != expected_dataset or not paths.dataset_root.is_dir():
        raise ValueError("dataset_root must be the existing immutable dataset/Main")
    output_root = paths.output_path(train_document["output_root"])
    if output_root != (project_root / "experiments" / "hosp_bonenet").resolve():
        raise ValueError("unsafe HOSP output_root")
    workspace = paths.output_path(data_document.get("workspace_root", "workspace"))
    if workspace != (project_root / "workspace").resolve():
        raise ValueError("workspace_root must be the canonical project workspace")
    schema_path = paths.output_path(data_document["schema_path"])
    if schema_path != (workspace / "annotations" / "label_schema.yaml").resolve():
        raise ValueError("schema_path must be the canonical label schema")
    schema = load_config(schema_path)
    _validate_schema(schema)
    split_dir = (workspace / "splits").resolve()
    for split in ("fit_train", "val"):
        if not split_dir.joinpath(f"{split}.csv").is_file():
            raise FileNotFoundError(f"missing required {split} manifest")
    inputs = _input_record(split_dir, schema_path)
    if project_root == PROJECT_ROOT:
        expected = {
            "split_generation_id": EXPECTED_SPLIT_GENERATION_ID,
            "manifest_sha256": {
                "fit_train": EXPECTED_INPUT_SHA256["fit_train"],
                "val": EXPECTED_INPUT_SHA256["val"],
            },
            "schema_sha256": EXPECTED_INPUT_SHA256["schema"],
        }
        if not _strict_equal(inputs, expected):
            raise ValueError("production manifests/schema do not match frozen Phase 7 inputs")

    runtime_data = {
        "dataset_root": str(paths.dataset_root),
        "image_size": train_document["image_size"],
        "seed": train_document["seed"],
        "preprocessing": dict(data_document.get("preprocessing", {})),
        "augmentation": dict(data_document.get("augmentation", {})),
        "loader": {
            "batch_size": train_document["microbatch_size"],
            "num_workers": train_document["num_workers"],
            "pin_memory": train_document["pin_memory"],
            "persistent_workers": train_document["persistent_workers"],
            "sampling": str(data_document.get("loader", {}).get("sampling", "shuffle")),
        },
    }
    record = {
        **dict(train_document),
        "model_name": "hosp_bonenet",
        "model": dict(model_document["model"]),
        "loss_weights": dict(model_document["loss_weights"]),
        "verified_supervision": ["binary"],
        "model_config": _relative(model_path, project_root),
        "train_config": _relative(train_path, project_root),
        "data_config": _relative(data_path, project_root),
        "dataset_root": _relative(paths.dataset_root, project_root),
        "split_dir": _relative(split_dir, project_root),
        "schema_path": _relative(schema_path, project_root),
        "selection": {"split": "val", "metric": "AUROC"},
        "inputs": inputs,
    }
    runtime = {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "split_dir": str(split_dir),
        "schema_path": str(schema_path),
        "schema": schema,
        "data": runtime_data,
    }
    return {"record": record, "runtime": runtime}


def select_final_stage(stage_results: Mapping[str, Mapping[str, Any]]) -> str:
    """Select higher finite validation AUROC, preferring earlier Stage A on a tie."""

    if set(stage_results) != {"A", "D"}:
        raise ValueError("stage results must contain exactly A and D")
    scores: dict[str, float] = {}
    for key in ("A", "D"):
        try:
            score = float(stage_results[key]["best_AUROC"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Stage {key} has no numeric best_AUROC") from error
        if not math.isfinite(score):
            raise ValueError(f"Stage {key} best_AUROC must be finite")
        scores[key] = score
    return "D" if scores["D"] > scores["A"] else "A"


def _stage_config(record: Mapping[str, Any], stage_key: str) -> dict[str, Any]:
    stage = record["stages"][stage_key]
    return {
        "epochs": stage["epochs"],
        "gradient_accumulation_steps": record["gradient_accumulation_steps"],
        "gradient_clip_norm": record["gradient_clip_norm"],
        "seed": record["seed"],
        "selection_metric": "AUROC",
        "learning_rate": stage["learning_rate"],
        "weight_decay": record["weight_decay"],
        "warmup_epochs": stage["warmup_epochs"],
        "early_stopping_patience": stage["early_stopping_patience"],
        "amp": record["amp"],
        "stage_key": stage_key,
        "stage_name": stage["name"],
        "monitor_extended": True,
    }


def _new_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = output_root / f"{stamp}-{uuid4().hex[:8]}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a unique HOSP run directory")


def _resume_directory(
    resume_run: Path, output_root: Path, record: Mapping[str, Any]
) -> tuple[Path, str]:
    run_dir = Path(resume_run).resolve()
    if run_dir.parent != output_root.resolve() or not run_dir.is_dir():
        raise ValueError("resume run must be an existing direct child of HOSP output_root")
    status_path = run_dir / "status.json"
    config_path = run_dir / "config.yaml"
    if not status_path.is_file() or not config_path.is_file():
        raise ValueError("resume run lacks status/config provenance")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    prior_status = status.get("status") if isinstance(status, Mapping) else None
    if prior_status not in {"failed", "running"}:
        raise ValueError("only a failed or interrupted run can be resumed")
    saved_record = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not _strict_equal(saved_record, record):
        raise ValueError("resume run config does not match the current resolved config")
    for stage_key in ("A", "D"):
        stage_dir = run_dir / f"stage_{stage_key}"
        if not stage_dir.exists():
            continue
        present = {path.name for path in stage_dir.iterdir() if path.is_file()}
        checkpoints = {"best.pt", "last.pt"} & present
        if checkpoints == {"best.pt"}:
            raise ValueError(f"stage_{stage_key} has an incomplete checkpoint pair")
    return run_dir, str(prior_status)


def _matching_run_status(run_dir: Path, record: Mapping[str, Any]) -> str | None:
    config_path = run_dir / "config.yaml"
    status_path = run_dir / "status.json"
    if not config_path.is_file() or not status_path.is_file():
        return None
    try:
        saved_record = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"run {run_dir.name} has malformed config provenance") from error
    if not _strict_equal(saved_record, record):
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"run {run_dir.name} has malformed status provenance") from error
    if not isinstance(status, Mapping) or not isinstance(status.get("status"), str):
        raise ValueError(f"run {run_dir.name} has malformed status provenance")
    return str(status["status"])


def _discover_exact_run(
    output_root: Path, record: Mapping[str, Any]
) -> tuple[str, Path] | None:
    if not output_root.is_dir():
        return None
    completed: list[Path] = []
    failed: list[Path] = []
    running: list[Path] = []
    for run_dir in sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        status = _matching_run_status(run_dir, record)
        if status == "complete":
            completed.append(run_dir)
        elif status == "failed" and any(
            (run_dir / f"stage_{stage_key}" / "last.pt").is_file()
            for stage_key in ("A", "D")
        ):
            failed.append(run_dir)
        elif status == "running":
            running.append(run_dir)
        elif status is None:
            continue
        elif status is not None:
            raise ValueError(f"run {run_dir.name} has unsupported status {status!r}")
    if completed:
        return "complete", completed[0]
    if running:
        return "running", running[0]
    if failed:
        return "failed", failed[0]
    return None


def _configuration_fingerprint(record: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _register_selected_checkpoint(
    run_dir: Path,
    record: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    project_root: Path,
    replace_alias: bool,
) -> None:
    from bone_cancer_ai.artifacts.registry import promote_checkpoint

    selected_checkpoint = run_dir / str(selection["selected_checkpoint"])
    selected_stage = str(selection["selected_stage"])
    promote_checkpoint(
        project_root,
        alias=SELECTED_CHECKPOINT_ALIAS,
        checkpoint_path=selected_checkpoint,
        replace_alias=replace_alias,
        model_family="hosp_bonenet",
        phase="phase_7",
        source_run=run_dir,
        selected_metric={
            "split": "val",
            "name": "AUROC",
            "value": float(selection["stage_scores"][selected_stage]),
        },
        configuration_fingerprint=_configuration_fingerprint(record),
        test_status={"status": "not_run", "planned_phase": 9},
    )


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
        *(f"{name}={version}" for name, version in packages.items()),
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda or 'unavailable'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_git_state(path: Path, project_root: Path) -> None:
    try:
        commit = subprocess.run(
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
        contents = f"commit={commit}\ndirty={str(dirty).lower()}\n"
    except (OSError, subprocess.CalledProcessError):
        contents = "commit=unavailable\ndirty=unavailable\n"
    path.write_text(contents, encoding="utf-8", newline="\n")


def _require_clean_git(project_root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("production training requires an available Git repository") from error
    if not commit or dirty:
        raise RuntimeError("production training requires a clean committed Git tree")
    return commit


def _write_notes(path: Path, *, status: str, detail: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"Status: {status}",
                "Stages: A binary global/local; B skipped; C skipped; D lower-LR fine-tune.",
                "Checkpoint selection: validation AUROC; exact ties prefer Stage A.",
                "Calibration and test are untouched. Test status: not_run; planned for Phase 9.",
                detail,
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _default_plot_history(history: list[dict[str, Any]], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("cannot plot an empty training history")
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Binary loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_AUROC"] for row in history])
    axes[1].set_title("Validation AUROC")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(figure)


def _default_dependencies() -> RunDependencies:
    import torch

    from bone_cancer_ai.data.datamodule import BoneCancerDataModule
    from bone_cancer_ai.evaluation.classification import binary_classification_metrics
    from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
    from bone_cancer_ai.training.checkpointing import load_checkpoint
    from bone_cancer_ai.training.optimizer import class_weight_from_frame
    from bone_cancer_ai.training.trainer import BinaryTrainer

    return RunDependencies(
        data_module_type=BoneCancerDataModule,
        trainer_type=BinaryTrainer,
        create_model=create_hosp_bonenet,
        class_weight=class_weight_from_frame,
        load_checkpoint=load_checkpoint,
        metrics=binary_classification_metrics,
        cuda_available=torch.cuda.is_available,
        plot_history=_default_plot_history,
    )


def _prediction_frame(prediction: Mapping[str, Any]) -> Any:
    import pandas as pd
    import torch

    logits = torch.as_tensor(prediction["logits"]).detach().cpu().to(torch.float64)
    targets = torch.as_tensor(prediction["targets"]).detach().cpu().to(torch.long)
    sample_ids = [str(value) for value in prediction["sample_ids"]]
    if logits.ndim != 1 or logits.shape != targets.shape or len(sample_ids) != len(logits):
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


def _same_prediction(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return bool(
        list(left["sample_ids"]) == list(right["sample_ids"])
        and torch.equal(torch.as_tensor(left["targets"]), torch.as_tensor(right["targets"]))
        and torch.equal(torch.as_tensor(left["logits"]), torch.as_tensor(right["logits"]))
    )


def run_hosp_training(
    model_config_path: Path,
    *,
    dependencies: RunDependencies | None = None,
    resume_run: Path | None = None,
    force_new_run: bool = False,
) -> Path:
    """Execute approved stages and persist one unique validation-only bundle."""

    if force_new_run and resume_run is not None:
        raise ValueError("force_new_run cannot be combined with resume_run")
    resolved = resolve_config(model_config_path)
    record, runtime = resolved["record"], resolved["runtime"]
    project_root = Path(runtime["project_root"])
    output_root = Path(runtime["output_root"])
    selected_resume = resume_run
    if resume_run is None and not force_new_run:
        discovered = _discover_exact_run(output_root, record)
        if discovered is not None:
            discovered_status, discovered_run = discovered
            if discovered_status == "complete":
                selection = validate_run_bundle(discovered_run)
                _register_selected_checkpoint(
                    discovered_run,
                    record,
                    selection,
                    project_root=project_root,
                    replace_alias=False,
                )
                print(f"phase7_reuse run_id={discovered_run.name}", flush=True)
                return discovered_run
            if discovered_status == "running":
                raise RuntimeError(
                    f"exact HOSP run {discovered_run.name} is already running"
                )
            selected_resume = discovered_run
    production_run = dependencies is None
    orchestration_commit = (
        _require_clean_git(project_root)
        if production_run
        else "dependency-injected-test"
    )
    dependencies = dependencies or _default_dependencies()
    if not dependencies.cuda_available():
        raise RuntimeError("Phase 7 HOSP training requires CUDA")
    validate_recorded_inputs(record, runtime)
    if selected_resume is not None:
        from bone_cancer_ai.utils.locking import exclusive_file_lock

        resume_candidate = Path(selected_resume).resolve()
        if resume_candidate.parent != output_root.resolve() or not resume_candidate.is_dir():
            raise ValueError("resume run must be an existing direct child of HOSP output_root")
        with exclusive_file_lock(resume_candidate / ".run.lock"):
            return _execute_hosp_training(
                record=record,
                runtime=runtime,
                project_root=project_root,
                output_root=output_root,
                dependencies=dependencies,
                selected_resume=resume_candidate,
                orchestration_commit=orchestration_commit,
                force_new_run=force_new_run,
            )
    return _execute_hosp_training(
        record=record,
        runtime=runtime,
        project_root=project_root,
        output_root=output_root,
        dependencies=dependencies,
        selected_resume=None,
        orchestration_commit=orchestration_commit,
        force_new_run=force_new_run,
    )


def _execute_hosp_training(
    *,
    record: Mapping[str, Any],
    runtime: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    dependencies: RunDependencies,
    selected_resume: Path | None,
    orchestration_commit: str,
    force_new_run: bool,
) -> Path:
    import pandas as pd

    from bone_cancer_ai.evaluation.reports import atomic_write_csv, atomic_write_json

    prior_status: str | None = None
    if selected_resume is None:
        run_dir = _new_run_directory(output_root)
    else:
        run_dir, prior_status = _resume_directory(selected_resume, output_root, record)
    notes_path = run_dir / "notes.md"
    try:
        if selected_resume is None:
            run_dir.joinpath("config.yaml").write_text(
                yaml.safe_dump(record, sort_keys=False), encoding="utf-8", newline="\n"
            )
            _write_environment(run_dir / "environment.txt")
            _write_git_state(run_dir / "git_commit.txt", Path(runtime["project_root"]))
            detail = "Run initialized."
        else:
            resume_path = run_dir / "resume_provenance.txt"
            with resume_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    f"resumed_at={datetime.now(UTC).isoformat()} "
                    f"prior_status={prior_status} "
                    f"orchestration_commit={orchestration_commit}\n"
                )
            detail = f"Resumed from prior status: {prior_status}."
        _write_notes(notes_path, status="Running", detail=detail)
        atomic_write_json(
            {"status": "running", "test": "not_run", "planned_phase": 9},
            run_dir / "status.json",
        )
        _set_reproducibility(int(record["seed"]))
        data_module = dependencies.data_module_type(
            Path(runtime["split_dir"]),
            runtime["data"],
            runtime["schema"],
            splits=("fit_train", "val"),
        )
        data_module.setup()
        fit_loader = data_module.loader("fit_train")
        val_loader = data_module.loader("val")
        pos_weight = dependencies.class_weight(data_module.frames["fit_train"])
        class_counts = (
            data_module.frames["fit_train"]["binary_label"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        stage_results: dict[str, dict[str, Any]] = {}
        stage_metrics: dict[str, dict[str, object]] = {}
        stage_checkpoints: dict[str, dict[str, Any]] = {}
        for stage_key in ("A", "D"):
            stage_dir = run_dir / f"stage_{stage_key}"
            stage_dir.mkdir(exist_ok=True)
            stage_config = _stage_config(record, stage_key)
            model = dependencies.create_model(record["model"], runtime["schema"])
            if stage_key == "A":
                trainable_parameters = sum(
                    int(parameter.numel())
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                print(
                    "phase7_start "
                    f"run_id={run_dir.name} device={record['device']} "
                    f"model=hosp_bonenet train_samples={len(data_module.frames['fit_train'])} "
                    f"val_samples={len(data_module.frames['val'])} "
                    f"class_0={class_counts.get(0, 0)} class_1={class_counts.get(1, 0)} "
                    f"trainable_parameters={trainable_parameters}",
                    flush=True,
                )
            if stage_key == "D":
                dependencies.load_checkpoint(
                    run_dir / "stage_A" / "best.pt",
                    model=model,
                    expected_model_alias="hosp_bonenet",
                    map_location=record["device"],
                    expected_resolved_config=_stage_config(record, "A"),
                    expected_selection_metric="AUROC",
                )
            trainer_kwargs = {
                "model_alias": "hosp_bonenet",
                "resolved_config": stage_config,
                "checkpoint_dir": stage_dir,
                "pos_weight": pos_weight,
                "device": record["device"],
            }
            if (stage_dir / "last.pt").is_file():
                trainer_kwargs["resume_from"] = stage_dir / "last.pt"
                print(
                    f"phase7_resume stage={stage_key} checkpoint=stage_{stage_key}/last.pt",
                    flush=True,
                )
            trainer = dependencies.trainer_type(model, **trainer_kwargs)
            result = trainer.fit(fit_loader, val_loader)
            history = [dict(row) for row in result["history"]]
            dependencies.load_checkpoint(
                stage_dir / "best.pt",
                model=model,
                expected_model_alias="hosp_bonenet",
                map_location=record["device"],
                expected_resolved_config=stage_config,
                expected_selection_metric="AUROC",
            )
            prediction = trainer.predict(val_loader)
            metrics = dependencies.metrics(
                prediction["logits"], prediction["targets"], threshold=0.5
            )
            fresh_model = dependencies.create_model(record["model"], runtime["schema"])
            dependencies.load_checkpoint(
                stage_dir / "best.pt",
                model=fresh_model,
                expected_model_alias="hosp_bonenet",
                map_location=record["device"],
                expected_resolved_config=stage_config,
                expected_selection_metric="AUROC",
            )
            reload_trainer = dependencies.trainer_type(
                fresh_model,
                model_alias="hosp_bonenet",
                resolved_config=stage_config,
                checkpoint_dir=run_dir / f"reload_{stage_key}_unused",
                pos_weight=pos_weight,
                device=record["device"],
            )
            reloaded_prediction = reload_trainer.predict(val_loader)
            if not _same_prediction(prediction, reloaded_prediction):
                raise RuntimeError(
                    f"Stage {stage_key} best checkpoint failed exact validation reload"
                )
            stage_results[stage_key] = dict(result)
            stage_metrics[stage_key] = metrics
            stage_checkpoints[stage_key] = {
                "best_epoch": int(result["best_epoch"]),
                "best_AUROC": float(result["best_AUROC"]),
                "best_sha256": _sha256(stage_dir / "best.pt"),
                "last_sha256": _sha256(stage_dir / "last.pt"),
                "reload_exact": True,
            }
            atomic_write_csv(pd.DataFrame(history), stage_dir / "history.csv")
            dependencies.plot_history(history, stage_dir / "training_curves.png")
            atomic_write_csv(
                _prediction_frame(prediction), stage_dir / "validation_predictions.csv"
            )

        selected_stage = select_final_stage(stage_results)
        selected_checkpoint = run_dir / f"stage_{selected_stage}" / "best.pt"

        checkpoint_relative = selected_checkpoint.relative_to(run_dir).as_posix()
        selection = {
            "split": "val",
            "metric": "AUROC",
            "tie_break": "earlier_stage",
            "selected_stage": selected_stage,
            "selected_epoch": int(stage_results[selected_stage]["best_epoch"]),
            "selected_checkpoint": checkpoint_relative,
            "checkpoint_sha256": _sha256(selected_checkpoint),
            "stage_scores": {key: float(stage_results[key]["best_AUROC"]) for key in ("A", "D")},
            "stage_checkpoints": stage_checkpoints,
            "skipped_stages": ["B", "C"],
            "reload_identity": "exact_logits_targets_sample_ids",
        }
        metrics_payload = {
            "model": "hosp_bonenet",
            "seed": record["seed"],
            "selection_split": "val",
            "validation": stage_metrics[selected_stage],
            "stages": {key: stage_metrics[key] for key in ("A", "D")},
            "test": {"status": "not_run", "planned_phase": 9},
        }
        atomic_write_json(selection, run_dir / "selection.json")
        atomic_write_json(metrics_payload, run_dir / "metrics.json")
        run_dir.joinpath("best_checkpoint_path.txt").write_text(
            checkpoint_relative + "\n", encoding="utf-8", newline="\n"
        )
        _write_notes(notes_path, status="Complete", detail="Artifacts validated.")
        validate_recorded_inputs(record, runtime)
        atomic_write_json(
            {"status": "complete", "test": "not_run", "planned_phase": 9},
            run_dir / "status.json",
        )
        validated_selection = validate_run_bundle(run_dir)
    except Exception as error:
        _write_notes(
            notes_path,
            status="Failed",
            detail=f"Failure: {type(error).__name__}: {error}",
        )
        atomic_write_json(
            {
                "status": "failed",
                "test": "not_run",
                "planned_phase": 9,
                "error_type": type(error).__name__,
            },
            run_dir / "status.json",
        )
        raise
    _register_selected_checkpoint(
        run_dir,
        record,
        validated_selection,
        project_root=project_root,
        replace_alias=force_new_run,
    )
    print(
        "phase7_selection "
        f"stage={selected_stage} epoch={selection['selected_epoch']} "
        f"val_AUROC={selection['stage_scores'][selected_stage]:.6f} "
        f"checkpoint={checkpoint_relative}",
        flush=True,
    )
    return run_dir


def validate_run_bundle(run_dir: Path) -> dict[str, Any]:
    """Validate a completed HOSP bundle without loading calibration or test data."""

    import pandas as pd
    import torch

    from bone_cancer_ai.evaluation.classification import binary_classification_metrics

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing run bundle: {root}")
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.name.casefold().startswith("test") or "calibration" in path.name.casefold())
    ]
    if forbidden:
        raise ValueError("run bundle contains a forbidden test artifact")
    present = {path.name for path in root.iterdir() if path.is_file()}
    missing = ROOT_REQUIRED_FILES - present
    if missing:
        raise ValueError(f"run bundle is missing root artifacts: {sorted(missing)}")
    for evidence in ("config.yaml", "environment.txt", "git_commit.txt", "notes.md"):
        if not root.joinpath(evidence).read_text(encoding="utf-8").strip():
            raise ValueError(f"run bundle has empty {evidence}")
    status = json.loads(root.joinpath("status.json").read_text(encoding="utf-8"))
    if status != {"status": "complete", "test": "not_run", "planned_phase": 9}:
        raise ValueError("run status is not complete with test not_run")
    config = yaml.safe_load(root.joinpath("config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("run config is malformed")
    project_root = root.parents[2]
    if project_root == PROJECT_ROOT:
        git_record = dict(
            line.split("=", 1)
            for line in root.joinpath("git_commit.txt").read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        if git_record.get("dirty") != "false" or len(git_record.get("commit", "")) != 40:
            raise ValueError("production run lacks clean recoverable Git provenance")
    split_dir = (project_root / str(config.get("split_dir", ""))).resolve()
    schema_path = (project_root / str(config.get("schema_path", ""))).resolve()
    if not split_dir.is_relative_to(project_root) or not schema_path.is_relative_to(project_root):
        raise ValueError("run config input paths escape the project")
    validate_recorded_inputs(
        config, {"split_dir": str(split_dir), "schema_path": str(schema_path)}
    )
    selection = json.loads(root.joinpath("selection.json").read_text(encoding="utf-8"))
    metrics = json.loads(root.joinpath("metrics.json").read_text(encoding="utf-8"))
    checkpoint_records = selection.get("stage_checkpoints")
    if not isinstance(checkpoint_records, Mapping) or set(checkpoint_records) != {"A", "D"}:
        raise ValueError("selection lacks both stage checkpoint records")
    required_history = {
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "val_AUROC",
        "val_AUPRC",
        "val_F1",
        "stage",
        "train_binary_loss",
        "val_binary_loss",
        "epoch_time_seconds",
        "peak_cuda_memory_bytes",
        "checkpoint_decision",
    }
    derived_scores: dict[str, dict[str, float]] = {}
    for stage_key in ("A", "D"):
        stage_dir = root / f"stage_{stage_key}"
        if not stage_dir.is_dir():
            raise ValueError(f"run bundle is missing stage_{stage_key}")
        stage_files = {path.name for path in stage_dir.iterdir() if path.is_file()}
        missing_stage = STAGE_REQUIRED_FILES - stage_files
        if missing_stage:
            raise ValueError(f"stage_{stage_key} is missing artifacts: {sorted(missing_stage)}")
        for artifact in ("best.pt", "last.pt", "training_curves.png"):
            if stage_dir.joinpath(artifact).stat().st_size == 0:
                raise ValueError(f"stage_{stage_key} has an empty {artifact}")
        history = pd.read_csv(stage_dir / "history.csv")
        predictions = pd.read_csv(stage_dir / "validation_predictions.csv")
        if history.empty or predictions.empty:
            raise ValueError(f"stage_{stage_key} reports must be nonempty")
        if not required_history <= set(history):
            raise ValueError(f"stage_{stage_key} history is missing required monitors")
        for column in required_history - {"stage", "val_F1", "checkpoint_decision"}:
            values = pd.to_numeric(history[column], errors="coerce")
            if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"stage_{stage_key} history has invalid {column}")
        f1_values = pd.to_numeric(history["val_F1"], errors="coerce").dropna()
        if not all(math.isfinite(float(value)) for value in f1_values):
            raise ValueError(f"stage_{stage_key} history has invalid val_F1")
        expected_stage_name = config["stages"][stage_key]["name"]
        if not history["stage"].eq(expected_stage_name).all():
            raise ValueError(f"stage_{stage_key} history has the wrong stage label")
        running_best = -math.inf
        expected_decisions = []
        for score in history["val_AUROC"]:
            improved = float(score) > running_best
            expected_decisions.append("best" if improved else "last")
            running_best = max(running_best, float(score))
        if history["checkpoint_decision"].tolist() != expected_decisions:
            raise ValueError(f"stage_{stage_key} checkpoint decisions are inconsistent")
        if predictions.columns.tolist() != [
            "sample_id",
            "target",
            "logit",
            "probability",
            "prediction",
        ]:
            raise ValueError(f"stage_{stage_key} validation predictions are malformed")
        if predictions.empty or predictions["sample_id"].duplicated().any():
            raise ValueError(f"stage_{stage_key} validation predictions have invalid IDs")
        logits = pd.to_numeric(predictions["logit"], errors="coerce")
        targets = pd.to_numeric(predictions["target"], errors="coerce")
        probabilities = pd.to_numeric(predictions["probability"], errors="coerce")
        if (
            logits.isna().any()
            or targets.isna().any()
            or probabilities.isna().any()
            or not targets.isin([0, 1]).all()
        ):
            raise ValueError(f"stage_{stage_key} validation predictions are invalid")
        expected_probabilities = torch.sigmoid(torch.tensor(logits.tolist(), dtype=torch.float64))
        if not torch.allclose(
            expected_probabilities,
            torch.tensor(probabilities.tolist(), dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"stage_{stage_key} probabilities do not match logits")
        recomputed = binary_classification_metrics(logits.tolist(), targets.tolist())
        stored_metrics = metrics.get("stages", {}).get(stage_key)
        if not isinstance(stored_metrics, Mapping):
            raise ValueError(f"stage_{stage_key} metrics are missing")
        for metric_name, actual in recomputed.items():
            expected = stored_metrics.get(metric_name)
            if isinstance(actual, float):
                if not isinstance(expected, int | float) or not math.isclose(
                    actual, float(expected), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"stage_{stage_key} metric {metric_name} is inconsistent")
            elif actual != expected:
                raise ValueError(f"stage_{stage_key} metric {metric_name} is inconsistent")
        stage_record = checkpoint_records[stage_key]
        if not isinstance(stage_record, Mapping) or stage_record.get("reload_exact") is not True:
            raise ValueError(f"stage_{stage_key} lacks exact fresh-model reload evidence")
        best_epoch = int(history.loc[history["val_AUROC"].idxmax(), "epoch"])
        best_score = float(history["val_AUROC"].max())
        if stage_record.get("best_epoch") != best_epoch or not math.isclose(
            float(stage_record.get("best_AUROC", -1.0)), best_score, abs_tol=1e-12
        ):
            raise ValueError(f"stage_{stage_key} best checkpoint selection is inconsistent")
        for kind in ("best", "last"):
            checkpoint_path = stage_dir / f"{kind}.pt"
            if stage_record.get(f"{kind}_sha256") != _sha256(checkpoint_path):
                raise ValueError(f"stage_{stage_key} {kind} checkpoint checksum mismatch")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            expected_epoch = best_epoch if kind == "best" else int(history["epoch"].iloc[-1])
            if (
                not isinstance(payload, Mapping)
                or payload.get("model_alias") != "hosp_bonenet"
                or payload.get("selection_metric") != "AUROC"
                or payload.get("epoch") != expected_epoch
                or not _strict_equal(
                    payload.get("resolved_config"), _stage_config(config, stage_key)
                )
            ):
                raise ValueError(f"stage_{stage_key} {kind} checkpoint metadata mismatch")
            if kind == "best" and not math.isclose(
                float(payload.get("selection_value", -1.0)), best_score, abs_tol=1e-12
            ):
                raise ValueError(f"stage_{stage_key} best checkpoint score mismatch")
        derived_scores[stage_key] = {"best_AUROC": best_score}
    if selection.get("split") != "val" or selection.get("metric") != "AUROC":
        raise ValueError("selection must use validation AUROC")
    if selection.get("selected_stage") not in {"A", "D"}:
        raise ValueError("selection has an invalid stage")
    if type(selection.get("selected_epoch")) is not int or selection["selected_epoch"] < 1:
        raise ValueError("selection has an invalid epoch")
    if selection.get("skipped_stages") != ["B", "C"]:
        raise ValueError("Stages B and C must be explicitly skipped")
    expected_stage = select_final_stage(derived_scores)
    if selection["selected_stage"] != expected_stage:
        raise ValueError("selected stage does not follow the validation AUROC policy")
    stage_scores = selection.get("stage_scores")
    if not isinstance(stage_scores, Mapping) or any(
        key not in stage_scores
        or not math.isclose(
            float(stage_scores[key]), derived_scores[key]["best_AUROC"], abs_tol=1e-12
        )
        for key in ("A", "D")
    ):
        raise ValueError("selection stage scores are inconsistent with histories")
    selected_record = checkpoint_records[expected_stage]
    if selection["selected_epoch"] != selected_record["best_epoch"]:
        raise ValueError("selected epoch is inconsistent with stage metadata")
    checkpoint_text = root.joinpath("best_checkpoint_path.txt").read_text(encoding="utf-8").strip()
    checkpoint_reference = Path(checkpoint_text)
    expected_reference = Path(f"stage_{selection['selected_stage']}") / "best.pt"
    if (
        checkpoint_reference.is_absolute()
        or checkpoint_reference != expected_reference
        or (root / checkpoint_reference).resolve().parent.parent != root
        or not (root / checkpoint_reference).is_file()
        or selection.get("selected_checkpoint") != checkpoint_reference.as_posix()
    ):
        raise ValueError("best checkpoint path is unsafe or inconsistent")
    if selection.get("checkpoint_sha256") != _sha256(root / checkpoint_reference):
        raise ValueError("selected checkpoint checksum does not match selection metadata")
    if selection["checkpoint_sha256"] != selected_record["best_sha256"]:
        raise ValueError("selected checkpoint checksum disagrees with stage metadata")
    if metrics.get("selection_split") != "val" or metrics.get("test") != {
        "status": "not_run",
        "planned_phase": 9,
    }:
        raise ValueError("metrics violate validation-only/test-not-run policy")
    if not isinstance(metrics.get("validation"), dict) or not isinstance(
        metrics.get("stages"), dict
    ):
        raise ValueError("metrics payload is incomplete")
    if not _strict_equal(metrics["validation"], metrics["stages"][expected_stage]):
        raise ValueError("selected validation metrics do not match the selected stage")
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "hosp_bonenet.yaml"
    )
    parser.add_argument("--validate-bundle", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--force-new-run", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_bundle is not None:
        if (
            args.config != PROJECT_ROOT / "configs" / "hosp_bonenet.yaml"
            or args.resume_run is not None
            or args.force_new_run
        ):
            parser.error("--validate-bundle cannot be combined with other run options")
        print(json.dumps(validate_run_bundle(args.validate_bundle), indent=2))
        return 0
    if args.force_new_run and args.resume_run is not None:
        parser.error("--force-new-run cannot be combined with --resume-run")
    print(
        run_hosp_training(
            args.config,
            resume_run=args.resume_run,
            force_new_run=args.force_new_run,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
