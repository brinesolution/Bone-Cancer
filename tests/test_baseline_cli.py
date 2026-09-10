"""Contract tests for the Phase 5 baseline command and run bundles."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from bone_cancer_ai.evaluation import reports as report_module

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "09_train_baseline.py"
SPEC = importlib.util.spec_from_file_location("baseline_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
baseline_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline_cli
SPEC.loader.exec_module(baseline_cli)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _project_config(tmp_path: Path, **common_updates: object) -> Path:
    root = tmp_path / "project"
    (root / "dataset" / "Main").mkdir(parents=True)
    (root / "workspace" / "splits").mkdir(parents=True)
    for split in ("fit_train", "calibration", "val", "test"):
        root.joinpath("workspace", "splits", f"{split}.csv").write_text(
            f"assigned_split\n{split}\n", encoding="utf-8"
        )
    _write_yaml(
        root / "workspace" / "annotations" / "label_schema.yaml",
        {"binary": {"enabled": True, "classes": {0: "normal", 1: "bone_tumour"}}},
    )
    _write_yaml(
        root / "configs" / "data.yaml",
        {
            "project_root": str(root),
            "dataset_root": str(root / "dataset" / "Main"),
            "workspace_root": "workspace",
            "schema_path": "workspace/annotations/label_schema.yaml",
            "image_size": 384,
            "seed": 42,
            "preprocessing": {"normalize": "imagenet", "force_grayscale": True},
            "augmentation": {"horizontal_flip": False, "vertical_flip": False},
            "loader": {"batch_size": 8, "num_workers": 0},
        },
    )
    common = {
        "data_config": "data.yaml",
        "output_root": "experiments/baselines",
        "image_size": 384,
        "batch_size": 8,
        "epochs": 40,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "amp": True,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 8,
        "seed": 42,
        "device": "cuda",
        "allow_cpu_training": False,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "selection_split": "val",
        "selection_metric": "AUROC",
    }
    common.update(common_updates)
    _write_yaml(root / "configs" / "train.yaml", common)
    model_config = root / "configs" / "baseline_densenet121.yaml"
    _write_yaml(model_config, {"model": "densenet121", "common_config": "train.yaml"})
    return model_config


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"device": "cpu"}, "device"),
        ({"selection_split": "test"}, "selection_split"),
        ({"selection_metric": "test_AUROC"}, "selection_metric"),
        ({"epochs": 0}, "epochs"),
        ({"output_root": "../../outside"}, "output_root"),
        ({"output_root": "artifacts/runs"}, "experiments/baselines"),
        ({"output_root": "dataset/Main/runs"}, "dataset/Main"),
        ({"mystery_setting": True}, "unknown"),
    ],
)
def test_config_resolution_rejects_unsafe_or_unfair_values(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    config = _project_config(tmp_path, **updates)

    with pytest.raises(ValueError, match=message):
        baseline_cli.resolve_config(config)


def test_config_paths_are_relative_to_the_declaring_files_and_overrides_are_recorded(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)

    resolved = baseline_cli.resolve_config(
        config,
        epochs_override=2,
        pretrained_override=False,
        limit_batches=1,
    )

    root = config.parents[1].resolve()
    runtime = resolved["runtime"]
    record = resolved["record"]
    assert runtime["project_root"] == str(root)
    assert runtime["data_config_path"] == str((root / "configs" / "data.yaml").resolve())
    assert runtime["split_dir"] == str((root / "workspace" / "splits").resolve())
    assert runtime["schema_path"] == str(
        (root / "workspace" / "annotations" / "label_schema.yaml").resolve()
    )
    assert record["model_config"] == "configs/baseline_densenet121.yaml"
    assert record["data_config"] == "configs/data.yaml"
    assert record["split_dir"] == "workspace/splits"
    assert record["epochs"] == 2
    assert record["pretrained"] is False
    assert record["limit_batches"] == 1
    assert record["smoke_overrides"] == {
        "epochs": 2,
        "pretrained": False,
        "limit_batches": 1,
    }


def test_default_config_is_a_cuda_amp_pretrained_full_run(tmp_path: Path) -> None:
    resolved = baseline_cli.resolve_config(_project_config(tmp_path))["record"]

    assert resolved["device"] == "cuda"
    assert resolved["amp"] is True
    assert resolved["pretrained"] is True
    assert resolved["limit_batches"] is None
    assert resolved["epochs"] == 40
    assert resolved["smoke_overrides"] == {}


@pytest.mark.parametrize("missing_key", ["project_root", "dataset_root", "schema_path"])
def test_data_config_requires_boundary_paths(tmp_path: Path, missing_key: str) -> None:
    config = _project_config(tmp_path)
    data_path = config.parent / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    del data[missing_key]
    _write_yaml(data_path, data)

    with pytest.raises(ValueError, match=missing_key):
        baseline_cli.resolve_config(config)


@pytest.mark.parametrize(
    ("data_update", "message"),
    [
        ({"project_root": "../.."}, "project_root"),
        ({"dataset_root": "dataset"}, "dataset/Main"),
        ({"schema_path": "../outside.yaml"}, "schema_path"),
        ({"project_root": ["not", "a", "path"]}, "project_root"),
        ({"loader": "not-a-mapping"}, "loader"),
        ({"api_token": "secret-value"}, "unknown data config"),
    ],
)
def test_data_config_rejects_unsafe_boundaries_types_and_unknown_keys(
    tmp_path: Path, data_update: dict[str, object], message: str
) -> None:
    config = _project_config(tmp_path)
    data_path = config.parent / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(data_update)
    _write_yaml(data_path, data)

    with pytest.raises((TypeError, ValueError), match=message):
        baseline_cli.resolve_config(config)


def test_resolved_record_omits_absolute_and_provenance_paths(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    data_path = config.parent / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["provenance"] = {
        "patient_export": str(tmp_path / "private" / "patient-123"),
    }
    _write_yaml(data_path, data)

    resolved = baseline_cli.resolve_config(config)
    serialized = yaml.safe_dump(resolved["record"], sort_keys=False)

    assert str(config.parents[1]) not in serialized
    assert "patient-123" not in serialized
    assert "provenance" not in serialized
    for key in ("project_root", "dataset_root", "schema_path", "split_dir", "output_root"):
        assert Path(resolved["runtime"][key]).is_absolute()


@pytest.mark.parametrize("missing", ["schema", "fit_train", "calibration", "val", "test"])
def test_config_resolution_requires_schema_and_all_workspace_splits(
    tmp_path: Path, missing: str
) -> None:
    config = _project_config(tmp_path)
    root = config.parents[1]
    path = (
        root / "workspace" / "annotations" / "label_schema.yaml"
        if missing == "schema"
        else root / "workspace" / "splits" / f"{missing}.csv"
    )
    path.unlink()

    with pytest.raises(FileNotFoundError, match="schema|split"):
        baseline_cli.resolve_config(config)


def test_unknown_model_fails_in_a_real_subprocess_before_training(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    _write_yaml(config, {"model": "resnet50", "common_config": "train.yaml"})

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config)],
        cwd=SCRIPT_PATH.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Unknown baseline" in completed.stderr
    assert not (config.parents[1] / "experiments").exists()


class _FakeDataModule:
    events: list[str] = []

    def __init__(self, split_dir, data_config, label_schema) -> None:
        assert Path(split_dir).name == "splits"
        assert data_config["loader"]["batch_size"] == 8
        assert label_schema["binary"]["enabled"] is True
        self.frames = {
            "fit_train": pd.DataFrame(
                {
                    "assigned_split": ["fit_train", "fit_train"],
                    "is_eligible": [True, True],
                    "binary_label": [0, 1],
                }
            )
        }

    def setup(self) -> None:
        self.events.append("setup")

    def loader(self, split: str):
        self.events.append(f"loader:{split}")
        return [split]


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, 0, 0, 0] * self.weight


class _FakeTrainer:
    events: list[str] = []

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.directory = Path(kwargs["checkpoint_dir"])

    def fit(self, train_loader, val_loader):
        assert list(train_loader) == ["fit_train"]
        assert list(val_loader) == ["val"]
        self.events.append("fit")
        self.model.weight.data.fill_(7.0)
        self.directory.joinpath("best.pt").write_bytes(b"best")
        self.directory.joinpath("last.pt").write_bytes(b"last")
        return {
            "best_epoch": 1,
            "best_AUROC": 1.0,
            "history": [{"epoch": 1, "train_loss": 0.2, "val_AUROC": 1.0}],
        }

    def predict(self, loader):
        split = next(iter(loader))
        self.events.append(f"predict:{split}")
        multiplier = 1.0 if split == "val" else 2.0
        magnitude = float(self.model.weight.detach()) * multiplier
        return {
            "logits": torch.tensor([-magnitude, magnitude]),
            "targets": torch.tensor([0, 1]),
            "sample_ids": [f"{split}-a", f"{split}-b"],
            "valid_count": 2,
            "rejected_count": 0,
        }


def _fake_dependencies(events: list[str]):
    _FakeDataModule.events = events
    _FakeTrainer.events = events

    def load_best(path, **kwargs):
        assert Path(path).name == "best.pt"
        kwargs["model"].weight.data.fill_(2.0)
        events.append("load_best")
        return {"epoch": 1}

    def metrics(logits, targets, threshold=0.5):
        del targets, threshold
        magnitude = float(torch.as_tensor(logits).abs().max())
        if magnitude == 2.0:
            score = 0.8
        elif magnitude == 4.0:
            score = 0.9
        else:
            raise AssertionError("predictions did not use the selected best checkpoint")
        return {
            "AUROC": score,
            "AUPRC": score - 0.1,
            "accuracy": score,
            "balanced_accuracy": score,
            "precision": score,
            "recall": score,
            "sensitivity": score,
            "specificity": score,
            "F1": score,
            "MCC": score - 0.2,
            "Brier": 1.0 - score,
            "ECE": 1.0 - score,
            "confusion_matrix": [[1, 0], [0, 1]],
        }

    return baseline_cli.RunDependencies(
        data_module_type=_FakeDataModule,
        trainer_type=_FakeTrainer,
        create_model=lambda **kwargs: _FakeModel(),
        count_parameters=lambda model: 1,
        class_weight=lambda frame: torch.tensor(1.0),
        load_checkpoint=load_best,
        metrics=metrics,
        latency=lambda *args, **kwargs: {
            "device": "cuda",
            "warmup": 1,
            "repetitions": 1,
            "total_elapsed_ms": 2.5,
            "latency_ms": 2.5,
        },
        cuda_available=lambda: True,
        cuda_peak_memory=lambda: 4096,
        cuda_reset_peak=lambda: None,
    )


def test_successful_runs_are_unique_complete_and_never_use_test_for_selection(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)
    events: list[str] = []
    dependencies = _fake_dependencies(events)

    first = baseline_cli.run_baseline(config, dependencies=dependencies)
    second = baseline_cli.run_baseline(config, dependencies=dependencies)

    assert first != second
    assert first.parent == second.parent
    required = {
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
    assert {path.name for path in first.iterdir()} == required
    assert events == [
        "setup",
        "loader:fit_train",
        "loader:val",
        "loader:test",
        "fit",
        "load_best",
        "predict:val",
        "predict:test",
    ] * 2
    metrics = json.loads(first.joinpath("metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation"]["AUROC"] == 0.8
    assert metrics["test"]["AUROC"] == 0.9
    assert metrics["selection"] == {"split": "val", "metric": "AUROC", "threshold": 0.5}
    assert "test is evaluated once" in first.joinpath("notes.md").read_text(encoding="utf-8")
    summary = pd.read_csv(config.parents[1] / "artifacts" / "metrics" / "baseline_summary.csv")
    assert summary.columns.tolist() == report_module.BASELINE_SUMMARY_COLUMNS
    assert len(summary) == 1
    assert summary.iloc[0]["AUROC"] == pytest.approx(0.9)
    predictions = pd.read_csv(first / "test_predictions.csv")
    assert predictions["logit"].tolist() == [-4.0, 4.0]


def test_failed_run_records_status_without_metrics_or_summary_row(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    dependencies = _fake_dependencies([])

    class FailingTrainer(_FakeTrainer):
        def fit(self, train_loader, val_loader):
            raise RuntimeError("synthetic fit failure")

    dependencies.trainer_type = FailingTrainer

    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        baseline_cli.run_baseline(config, dependencies=dependencies)

    runs = list((config.parents[1] / "experiments" / "baselines" / "densenet121").iterdir())
    assert len(runs) == 1
    assert "Status: Failed" in runs[0].joinpath("notes.md").read_text(encoding="utf-8")
    assert not runs[0].joinpath("metrics.json").exists()
    assert not (config.parents[1] / "artifacts" / "metrics" / "baseline_summary.csv").exists()


def test_post_allocation_initialization_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _project_config(tmp_path)

    def fail_environment(path: Path) -> None:
        del path
        raise RuntimeError("synthetic environment failure")

    monkeypatch.setattr(baseline_cli, "_write_environment", fail_environment)
    with pytest.raises(RuntimeError, match="synthetic environment failure"):
        baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))

    run = next((config.parents[1] / "experiments" / "baselines" / "densenet121").iterdir())
    assert "Status: Failed" in run.joinpath("notes.md").read_text(encoding="utf-8")


def test_cuda_preflight_happens_before_a_run_directory_is_created(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    dependencies = _fake_dependencies([])
    dependencies.cuda_available = lambda: False

    with pytest.raises(RuntimeError, match="CUDA"):
        baseline_cli.run_baseline(config, dependencies=dependencies)

    assert not (config.parents[1] / "experiments").exists()


def test_cuda_oom_adds_actionable_guidance_without_fake_results(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    dependencies = _fake_dependencies([])

    class OomTrainer(_FakeTrainer):
        def fit(self, train_loader, val_loader):
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")

    dependencies.trainer_type = OomTrainer

    with pytest.raises(torch.cuda.OutOfMemoryError, match="gradient accumulation"):
        baseline_cli.run_baseline(config, dependencies=dependencies)

    run = next((config.parents[1] / "experiments" / "baselines" / "densenet121").iterdir())
    notes = run.joinpath("notes.md").read_text(encoding="utf-8")
    assert "Status: Failed" in notes
    assert "batch size" in notes
    assert "gradient accumulation" in notes
    assert "resolution" in notes
    assert not run.joinpath("metrics.json").exists()
    assert not (config.parents[1] / "artifacts" / "metrics" / "baseline_summary.csv").exists()


def test_smoke_run_is_recorded_only_in_its_bundle_not_the_final_summary(tmp_path: Path) -> None:
    config = _project_config(tmp_path)

    run = baseline_cli.run_baseline(
        config,
        epochs_override=1,
        pretrained_override=False,
        limit_batches=1,
        dependencies=_fake_dependencies([]),
    )

    resolved = yaml.safe_load(run.joinpath("config.yaml").read_text(encoding="utf-8"))
    assert resolved["smoke_overrides"] == {
        "epochs": 1,
        "pretrained": False,
        "limit_batches": 1,
    }
    assert "Run type: smoke" in run.joinpath("notes.md").read_text(encoding="utf-8")
    assert not (config.parents[1] / "artifacts" / "metrics" / "baseline_summary.csv").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("epochs", 20),
        ("early_stopping_patience", 7),
        ("image_size", 512),
        ("batch_size", 4),
        ("gradient_accumulation_steps", 2),
        ("learning_rate", 0.0002),
        ("weight_decay", 0.0002),
        ("warmup_epochs", 1),
        ("gradient_clip_norm", 2.0),
        ("seed", 43),
    ],
)
def test_valid_yaml_policy_deviation_is_marked_nonfinal(
    tmp_path: Path, key: str, value: object
) -> None:
    record = baseline_cli.resolve_config(_project_config(tmp_path, **{key: value}))["record"]

    assert record["run_type"] == "smoke"
    assert baseline_cli.is_final_run_config(record) is False


def test_valid_yaml_policy_deviation_cannot_publish_summary(tmp_path: Path) -> None:
    config = _project_config(tmp_path, learning_rate=0.0002)

    run = baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))

    persisted = yaml.safe_load(run.joinpath("config.yaml").read_text(encoding="utf-8"))
    assert persisted["run_type"] == "smoke"
    assert not (config.parents[1] / "artifacts" / "metrics" / "baseline_summary.csv").exists()


def test_summary_update_replaces_model_seed_atomically_and_sorts(tmp_path: Path) -> None:
    destination = tmp_path / "baseline_summary.csv"
    base = {
        "seed": 42,
        "best_epoch": 1,
        "AUROC": 0.7,
        "AUPRC": 0.6,
        "balanced_accuracy": 0.5,
        "F1": 0.5,
        "sensitivity": 0.5,
        "specificity": 0.5,
        "MCC": 0.0,
        "params": 10,
        "latency_ms": 1.0,
    }
    report_module.update_baseline_summary(destination, {"model": "swin_tiny", **base})
    report_module.update_baseline_summary(destination, {"model": "densenet121", **base})
    report_module.update_baseline_summary(
        destination, {"model": "densenet121", **base, "AUROC": 0.9, "best_epoch": 2}
    )

    frame = pd.read_csv(destination)
    assert frame.columns.tolist() == report_module.BASELINE_SUMMARY_COLUMNS
    assert frame["model"].tolist() == ["densenet121", "swin_tiny"]
    assert frame.iloc[0]["AUROC"] == pytest.approx(0.9)
    assert frame.iloc[0]["best_epoch"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_phase5_validator_requires_four_complete_final_runs(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    dependencies = _fake_dependencies([])
    first = baseline_cli.run_baseline(config, dependencies=dependencies)

    with pytest.raises(ValueError, match="four approved"):
        baseline_cli.validate_phase5_artifacts(config.parents[1])

    assert first.is_dir()


def test_phase5_validator_accepts_four_complete_final_runs(tmp_path: Path) -> None:
    first_config = _project_config(tmp_path)
    config_dir = first_config.parent
    for alias in ("densenet121", "efficientnet_b3", "convnext_tiny", "swin_tiny"):
        config = config_dir / f"baseline_{alias}.yaml"
        _write_yaml(config, {"model": alias, "common_config": "train.yaml"})
        baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))

    result = baseline_cli.validate_phase5_artifacts(first_config.parents[1])

    assert result == {"models": 4, "runs": 4, "status": "valid"}


def test_phase5_validator_rejects_a_smoke_override_bundle(tmp_path: Path) -> None:
    first_config = _project_config(tmp_path)
    config_dir = first_config.parent
    for alias in ("densenet121", "efficientnet_b3", "convnext_tiny", "swin_tiny"):
        config = config_dir / f"baseline_{alias}.yaml"
        _write_yaml(config, {"model": alias, "common_config": "train.yaml"})
        kwargs = {"epochs_override": 1} if alias == "swin_tiny" else {}
        baseline_cli.run_baseline(
            config,
            dependencies=_fake_dependencies([]),
            **kwargs,
        )

    with pytest.raises(ValueError, match="four approved"):
        baseline_cli.validate_phase5_artifacts(first_config.parents[1])


def test_phase5_validator_keeps_latest_final_when_a_newer_smoke_run_exists(
    tmp_path: Path,
) -> None:
    first_config = _project_config(tmp_path)
    config_dir = first_config.parent
    for alias in ("densenet121", "efficientnet_b3", "convnext_tiny", "swin_tiny"):
        config = config_dir / f"baseline_{alias}.yaml"
        _write_yaml(config, {"model": alias, "common_config": "train.yaml"})
        baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))
    baseline_cli.run_baseline(
        config_dir / "baseline_swin_tiny.yaml",
        epochs_override=1,
        dependencies=_fake_dependencies([]),
    )

    assert baseline_cli.validate_phase5_artifacts(first_config.parents[1])["status"] == "valid"


def _four_final_runs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    first_config = _project_config(tmp_path)
    runs = {}
    for alias in ("densenet121", "efficientnet_b3", "convnext_tiny", "swin_tiny"):
        config = first_config.parent / f"baseline_{alias}.yaml"
        _write_yaml(config, {"model": alias, "common_config": "train.yaml"})
        runs[alias] = baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))
    return first_config.parents[1], runs


def test_validator_rejects_stale_summary_instead_of_matching_an_older_run(
    tmp_path: Path,
) -> None:
    root, runs = _four_final_runs(tmp_path)
    config = root / "configs" / "baseline_densenet121.yaml"
    newer = baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))
    assert newer.name > runs["densenet121"].name
    metrics = json.loads(newer.joinpath("metrics.json").read_text(encoding="utf-8"))
    metrics["test"]["AUROC"] = 0.123
    newer.joinpath("metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="summary AUROC"):
        baseline_cli.validate_phase5_artifacts(root)


def test_validator_requires_nonempty_evidence_and_exact_complete_status(tmp_path: Path) -> None:
    root, runs = _four_final_runs(tmp_path)
    run = runs["swin_tiny"]
    cases = [
        ("environment.txt", ""),
        ("git_commit.txt", ""),
        ("notes.md", "Status: Running\n"),
        ("notes.md", "Status: Complete\nStatus: Failed\n"),
    ]
    originals = {
        name: run.joinpath(name).read_text(encoding="utf-8") for name, _ in cases
    }
    for name, bad_content in cases:
        run.joinpath(name).write_text(bad_content, encoding="utf-8")
        with pytest.raises(ValueError, match="complete final run|evidence|Status"):
            baseline_cli.validate_phase5_artifacts(root)
        run.joinpath(name).write_text(originals[name], encoding="utf-8")


def test_validator_rejects_summary_link_to_nonfinal_bundle(tmp_path: Path) -> None:
    root, runs = _four_final_runs(tmp_path)
    config_path = runs["swin_tiny"] / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["learning_rate"] = 0.0002
    config["run_type"] = "smoke"
    _write_yaml(config_path, config)

    with pytest.raises(ValueError, match="complete final run|final policy"):
        baseline_cli.validate_phase5_artifacts(root)


def test_validator_rejects_tampered_embedded_data_metadata(tmp_path: Path) -> None:
    root, runs = _four_final_runs(tmp_path)
    config_path = runs["swin_tiny"] / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["seed"] = 99
    _write_yaml(config_path, config)

    with pytest.raises(ValueError, match="complete final run|sanitized"):
        baseline_cli.validate_phase5_artifacts(root)


def test_persisted_run_config_contains_only_sanitized_relative_paths(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    data_path = config.parent / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["provenance"] = {"private_source": str(tmp_path / "patient-secret")}
    _write_yaml(data_path, data)

    run = baseline_cli.run_baseline(config, dependencies=_fake_dependencies([]))
    persisted = run.joinpath("config.yaml").read_text(encoding="utf-8")

    assert str(config.parents[1]) not in persisted
    assert "patient-secret" not in persisted
    assert "provenance" not in persisted


def test_phase5_commands_expose_four_trainers_and_validation() -> None:
    project_root = SCRIPT_PATH.parents[1]
    tasks = json.loads(project_root.joinpath(".vscode/tasks.json").read_text(encoding="utf-8"))
    by_label = {task["label"]: task for task in tasks["tasks"]}
    expected = {
        "Train baseline: DenseNet121": "baseline_densenet121.yaml",
        "Train baseline: EfficientNet-B3": "baseline_efficientnet_b3.yaml",
        "Train baseline: ConvNeXt-Tiny": "baseline_convnext_tiny.yaml",
        "Train baseline: Swin-Tiny": "baseline_swin_tiny.yaml",
    }
    for label, config_name in expected.items():
        assert by_label[label]["command"].endswith("/.venv/Scripts/python.exe")
        assert by_label[label]["args"] == [
            "scripts/09_train_baseline.py",
            "--config",
            f"configs/{config_name}",
        ]
    assert by_label["Validate Phase 5 artifacts"]["args"] == [
        "scripts/09_train_baseline.py",
        "--validate-artifacts",
    ]

    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    assert powershell is not None
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/run_phase.ps1",
            "-Action",
            "phase5",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "four approved" in completed.stderr
