"""Train one approved Phase 5 baseline or validate completed run bundles."""

from __future__ import annotations

import argparse
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
    path = Path(str(value))
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


def resolve_config(
    model_config_path: Path,
    *,
    epochs_override: int | None = None,
    pretrained_override: bool | None = None,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    """Resolve and validate a model/common/data configuration chain."""

    model_path = Path(model_config_path).resolve()
    model_config = _load_yaml(model_path, name="model config")
    unknown_model_keys = set(model_config) - {"model", "common_config"}
    if unknown_model_keys:
        raise ValueError(f"unknown model config keys: {sorted(unknown_model_keys)}")
    model = model_config.get("model")
    if model not in APPROVED_MODELS:
        approved = ", ".join(APPROVED_MODELS)
        raise ValueError(f"Unknown baseline {model!r}; choose one of: {approved}")
    common_path = _resolve_declared_path(model_config.get("common_config"), model_path)
    common = _load_yaml(common_path, name="common config")
    unknown_common_keys = set(common) - COMMON_CONFIG_KEYS
    if unknown_common_keys:
        raise ValueError(f"unknown common config keys: {sorted(unknown_common_keys)}")
    missing_common_keys = COMMON_CONFIG_KEYS - set(common)
    if missing_common_keys:
        raise ValueError(f"common config is missing keys: {sorted(missing_common_keys)}")

    data_path = _resolve_declared_path(common["data_config"], common_path)
    data_config = _load_yaml(data_path, name="data config")
    if "project_root" not in data_config:
        raise ValueError("data config must define project_root")
    project_root = _resolve_declared_path(data_config["project_root"], data_path)
    _require_within(model_path, project_root, name="model config")
    _require_within(common_path, project_root, name="common config")
    _require_within(data_path, project_root, name="data config")

    output_root = (project_root / str(common["output_root"])).resolve()
    _require_within(output_root, project_root, name="output_root")
    dataset_root = (project_root / str(data_config.get("dataset_root", ""))).resolve()
    _require_within(dataset_root, project_root, name="dataset_root")
    source_root = (project_root / "dataset" / "Main").resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("output_root must not be inside dataset/Main")
    expected_output_root = (project_root / "experiments" / "baselines").resolve()
    if output_root != expected_output_root:
        raise ValueError("output_root must be experiments/baselines")
    workspace_root = (project_root / str(data_config.get("workspace_root", "workspace"))).resolve()
    _require_within(workspace_root, project_root, name="workspace_root")
    schema_path = (project_root / str(data_config.get("schema_path", ""))).resolve()
    _require_within(schema_path, project_root, name="schema_path")
    split_dir = (workspace_root / "splits").resolve()
    _require_within(split_dir, project_root, name="split_dir")

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
    if common["image_size"] != 384:
        raise ValueError("image_size must remain 384")
    for key in (
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

    resolved = dict(common)
    resolved["model"] = model
    resolved["model_config_path"] = str(model_path)
    resolved["common_config_path"] = str(common_path)
    resolved["data_config_path"] = str(data_path)
    resolved["project_root"] = str(project_root)
    resolved["output_root"] = str(output_root)
    resolved["dataset_root"] = str(dataset_root)
    resolved["split_dir"] = str(split_dir)
    resolved["schema_path"] = str(schema_path)
    resolved["pretrained"] = True if pretrained_override is None else pretrained_override
    resolved["limit_batches"] = limit_batches
    smoke_overrides: dict[str, object] = {}
    if epochs_override is not None:
        resolved["epochs"] = epochs_override
        resolved["warmup_epochs"] = min(int(resolved["warmup_epochs"]), epochs_override)
        smoke_overrides["epochs"] = epochs_override
    if pretrained_override is not None:
        smoke_overrides["pretrained"] = pretrained_override
    if limit_batches is not None:
        smoke_overrides["limit_batches"] = limit_batches
    resolved["smoke_overrides"] = smoke_overrides

    loader = dict(data_config.get("loader", {}))
    loader.update(
        {
            "batch_size": resolved["batch_size"],
            "num_workers": resolved["num_workers"],
            "pin_memory": resolved["pin_memory"],
            "persistent_workers": resolved["persistent_workers"],
        }
    )
    resolved_data = dict(data_config)
    resolved_data.update(
        {
            "project_root": str(project_root),
            "dataset_root": str(dataset_root),
            "image_size": resolved["image_size"],
            "seed": resolved["seed"],
            "loader": loader,
        }
    )
    resolved["data"] = resolved_data
    return resolved


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
    run_kind = "smoke" if overrides else "final"
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

    project_root = Path(resolved["project_root"])
    run_dir = _new_run_directory(Path(resolved["output_root"]), str(resolved["model"]))
    notes_path = run_dir / "notes.md"
    run_dir.joinpath("config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8", newline="\n"
    )
    _write_environment(run_dir / "environment.txt")
    _write_git_state(run_dir / "git_commit.txt", project_root)
    _write_notes(notes_path, resolved, status="Running", detail="Run initialized.")

    try:
        _set_reproducibility(int(resolved["seed"]))
        schema = _load_yaml(Path(resolved["schema_path"]), name="label schema")
        data_module = dependencies.data_module_type(
            Path(resolved["split_dir"]), resolved["data"], schema
        )
        data_module.setup()
        train_loader = _limit_loader(
            data_module.loader("fit_train"), resolved["limit_batches"]
        )
        val_loader = _limit_loader(data_module.loader("val"), resolved["limit_batches"])
        test_loader = _limit_loader(data_module.loader("test"), resolved["limit_batches"])
        model = dependencies.create_model(
            name=resolved["model"],
            pretrained=resolved["pretrained"],
            image_size=resolved["image_size"],
        )
        parameter_count = int(dependencies.count_parameters(model))
        pos_weight = dependencies.class_weight(data_module.frames["fit_train"])
        trainer = dependencies.trainer_type(
            model,
            model_alias=resolved["model"],
            resolved_config=resolved,
            checkpoint_dir=run_dir,
            pos_weight=pos_weight,
            device=resolved["device"],
        )
        dependencies.cuda_reset_peak()
        fit_result = trainer.fit(train_loader, val_loader)
        best_path = run_dir / "best.pt"
        dependencies.load_checkpoint(
            best_path,
            model=model,
            expected_model_alias=resolved["model"],
            map_location=resolved["device"],
            expected_resolved_config=resolved,
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
            torch.zeros(1, 3, int(resolved["image_size"]), int(resolved["image_size"])),
            resolved["device"],
        )
        resources = {
            "trainable_parameters": parameter_count,
            "best_checkpoint_bytes": best_path.stat().st_size,
            "cuda_peak_memory_bytes": int(dependencies.cuda_peak_memory()),
            "latency": latency,
        }
        metrics_payload = {
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
        atomic_write_csv(pd.DataFrame(fit_result["history"]), run_dir / "history.csv")
        atomic_write_csv(
            confusion_matrix_frame(test_metrics), run_dir / "confusion_matrix.csv"
        )
        atomic_write_csv(_prediction_frame(test_prediction), run_dir / "test_predictions.csv")
        atomic_write_json(metrics_payload, run_dir / "metrics.json")
        run_dir.joinpath("best_checkpoint_path.txt").write_text(
            "best.pt\n", encoding="utf-8", newline="\n"
        )
        _write_notes(notes_path, resolved, status="Complete", detail="Artifacts are complete.")
        summary_row = {
            "model": resolved["model"],
            "seed": resolved["seed"],
            "best_epoch": int(fit_result["best_epoch"]),
            **{column: test_metrics[column] for column in BASELINE_SUMMARY_COLUMNS[3:10]},
            "params": parameter_count,
            "latency_ms": latency["latency_ms"],
        }
        if not resolved["smoke_overrides"]:
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
        _write_notes(notes_path, resolved, status="Failed", detail=guidance)
        raise torch.cuda.OutOfMemoryError(f"{error}. {guidance}") from error
    except Exception as error:
        _remove_result_files(run_dir)
        detail = f"{type(error).__name__}: {str(error).splitlines()[0][:500]}"
        _write_notes(notes_path, resolved, status="Failed", detail=detail)
        raise


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _validate_run_bundle(run_dir: Path, alias: str, summary_row: Any) -> None:
    import pandas as pd

    present = {path.name for path in run_dir.iterdir() if path.is_file()}
    missing = REQUIRED_RUN_FILES - present
    if missing:
        raise ValueError(f"run {run_dir.name} is missing artifacts: {sorted(missing)}")
    config = _load_yaml(run_dir / "config.yaml", name="resolved run config")
    if config.get("model") != alias or config.get("seed") != 42:
        raise ValueError(f"run {run_dir.name} has the wrong model or seed")
    if config.get("smoke_overrides"):
        raise ValueError(f"run {run_dir.name} contains a smoke override")
    if not (
        config.get("pretrained") is True
        and config.get("device") == "cuda"
        and config.get("amp") is True
        and config.get("limit_batches") is None
    ):
        raise ValueError(f"run {run_dir.name} is not a final CUDA AMP pretrained run")
    if config.get("selection_split") != "val" or config.get("selection_metric") != "AUROC":
        raise ValueError(f"run {run_dir.name} contains test-selection violation evidence")
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
        validation_errors = []
        for candidate in candidates:
            try:
                _validate_run_bundle(candidate, alias, row)
            except (OSError, ValueError, KeyError) as error:
                validation_errors.append(str(error))
                continue
            break
        else:
            detail = validation_errors[0] if validation_errors else "no run directory exists"
            raise ValueError(f"no complete final run exists for {alias}: {detail}")
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
