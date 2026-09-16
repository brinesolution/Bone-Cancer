from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import yaml
from torch import nn

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "10_train_hosp_bonenet.py"
SPEC = importlib.util.spec_from_file_location("hosp_training_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
hosp_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hosp_cli)


def _training_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "output_root": "experiments/hosp_bonenet",
        "image_size": 384,
        "microbatch_size": 4,
        "effective_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "amp": True,
        "seed": 42,
        "device": "cuda",
        "allow_cpu_training": False,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "selection_split": "val",
        "selection_metric": "AUROC",
        "stages": {
            "A": {
                "enabled": True,
                "name": "A_binary_global_local",
                "epochs": 40,
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "warmup_epochs": 2,
                "early_stopping_patience": 8,
            },
            "B": {"enabled": False, "reason": "spatial supervision is not verified"},
            "C": {"enabled": False, "reason": "optional labels are not released"},
            "D": {
                "enabled": True,
                "name": "D_low_lr_finetune",
                "epochs": 10,
                "optimizer": "adamw",
                "learning_rate": 0.00001,
                "warmup_epochs": 0,
                "early_stopping_patience": 4,
                "initialize_from": "A.best",
            },
        },
    }
    document.update(overrides)
    return document


def _project_config(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    (project / "dataset" / "Main").mkdir(parents=True)
    split_dir = project / "workspace" / "splits"
    split_dir.mkdir(parents=True)
    for split in ("fit_train", "val"):
        split_dir.joinpath(f"{split}.csv").write_text(
            "sample_id,split_generation_id\n"
            f"{split}-sample,unit-generation\n",
            encoding="utf-8",
        )
    schema_dir = project / "workspace" / "annotations"
    schema_dir.mkdir(parents=True)
    schema = {
        "binary": {
            "enabled": True,
            "classes": {0: "normal", 1: "bone_tumour"},
            "semantics_verified": True,
        },
        "subtype": {"enabled": False},
        "risk": {"enabled": False},
        "concepts": {"enabled": False},
        "localization": {"boxes_verified": False, "masks_verified": False},
    }
    schema_dir.joinpath("label_schema.yaml").write_text(
        yaml.safe_dump(schema, sort_keys=False), encoding="utf-8"
    )
    data = {
        "project_root": str(project),
        "dataset_root": "dataset/Main",
        "source_read_only": True,
        "workspace_root": "workspace",
        "schema_path": "workspace/annotations/label_schema.yaml",
        "image_size": 384,
        "seed": 42,
        "preprocessing": {"force_grayscale": True, "normalize": "imagenet"},
        "augmentation": {},
        "loader": {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
            "sampling": "shuffle",
        },
    }
    configs.joinpath("data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    configs.joinpath("hosp_train.yaml").write_text(
        yaml.safe_dump(_training_document(), sort_keys=False), encoding="utf-8"
    )
    model = {
        "status": "phase_6_verified",
        "train_config": "hosp_train.yaml",
        "data_config": "data.yaml",
        "model": {
            "name": "hosp_bonenet",
            "backbone": "convnext_tiny",
            "pretrained": True,
            "feature_dim": 256,
            "pyramid_channels": 128,
            "dropout": 0.2,
            "pooling": "adaptive_avg",
            "localization_enabled": True,
            "subtype_enabled": False,
            "num_subtypes": 0,
            "risk_enabled": False,
            "num_risk_classes": 0,
            "concepts_enabled": False,
            "num_concepts": 0,
        },
        "loss_weights": {
            "binary": 1.0,
            "subtype": 0.0,
            "risk": 0.0,
            "segmentation": 0.0,
            "concepts": 0.0,
            "grounding": 0.0,
            "hierarchy": 0.0,
        },
        "verified_supervision": ["binary"],
    }
    path = configs / "hosp_bonenet.yaml"
    path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return path


def test_production_config_resolves_frozen_binary_two_stage_policy() -> None:
    resolved = hosp_cli.resolve_config(ROOT / "configs" / "hosp_bonenet.yaml")
    record = resolved["record"]

    assert record["microbatch_size"] == 4
    assert record["gradient_accumulation_steps"] == 2
    assert record["effective_batch_size"] == 8
    assert record["stages"]["A"] == {
        "enabled": True,
        "name": "A_binary_global_local",
        "epochs": 40,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "warmup_epochs": 2,
        "early_stopping_patience": 8,
    }
    assert record["stages"]["D"]["learning_rate"] == 0.00001
    assert record["stages"]["D"]["epochs"] == 10
    assert record["stages"]["B"]["enabled"] is False
    assert record["stages"]["C"]["enabled"] is False
    assert record["selection"] == {"split": "val", "metric": "AUROC"}
    assert record["inputs"]["split_generation_id"] == (
        "de64a489fff7202ec39d442e4b26a5a5c0d7e2c7db8ed7a8587601ef1cd28590"
    )
    assert set(record["inputs"]["manifest_sha256"]) == {"fit_train", "val"}
    assert len(record["inputs"]["schema_sha256"]) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"device": "cpu"}, "device.*cuda"),
        ({"selection_split": "test"}, "selection_split.*val"),
        ({"effective_batch_size": 7}, "effective batch"),
        ({"stages": {"A": {}, "D": {}}}, "stages"),
    ],
)
def test_training_policy_rejects_unsafe_or_unapproved_modes(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        hosp_cli.validate_training_policy(_training_document(**change))


def test_model_policy_rejects_optional_heads_and_losses() -> None:
    model = yaml.safe_load((ROOT / "configs" / "hosp_bonenet.yaml").read_text())
    model["model"]["concepts_enabled"] = True
    with pytest.raises(ValueError, match="optional.*disabled"):
        hosp_cli.validate_binary_model_policy(model)

    model["model"]["concepts_enabled"] = False
    model["loss_weights"]["concepts"] = 0.1
    with pytest.raises(ValueError, match="optional.*loss"):
        hosp_cli.validate_binary_model_policy(model)


def test_final_selector_prefers_stage_a_on_an_exact_tie() -> None:
    results = {"A": {"best_AUROC": 0.91}, "D": {"best_AUROC": 0.91}}
    assert hosp_cli.select_final_stage(results) == "A"
    results["D"]["best_AUROC"] = 0.92
    assert hosp_cli.select_final_stage(results) == "D"


def test_default_dependencies_reuse_shared_training_components() -> None:
    dependencies = hosp_cli._default_dependencies()
    assert dependencies.trainer_type.__name__ == "BinaryTrainer"
    assert dependencies.class_weight.__name__ == "class_weight_from_frame"


class _FakeDataModule:
    events: list[str] = []

    def __init__(self, split_dir, data_config, label_schema, *, splits) -> None:
        del split_dir, data_config, label_schema
        if tuple(splits) != ("fit_train", "val"):
            raise AssertionError(f"forbidden splits constructed: {splits}")
        self.frames = {
            "fit_train": pd.DataFrame(
                {
                    "assigned_split": ["fit_train", "fit_train"],
                    "is_eligible": [True, True],
                    "binary_label": [0, 1],
                }
            ),
            "val": pd.DataFrame({"binary_label": [0, 1]}),
        }

    def setup(self) -> None:
        self.events.append("setup")

    def loader(self, split: str):
        if split not in {"fit_train", "val"}:
            raise AssertionError(f"forbidden split accessed: {split}")
        self.events.append(f"loader:{split}")
        return [split]


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class _FakeTrainer:
    events: list[Any] = []

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.directory = Path(kwargs["checkpoint_dir"])
        self.config = kwargs["resolved_config"]
        resume_from = kwargs.get("resume_from")
        if resume_from is not None:
            payload = torch.load(resume_from, weights_only=True)
            self.model.weight.data.fill_(payload["weight"])
            self.events.append(("resume", self.config["stage_key"], Path(resume_from).name))
        self.events.append(
            (
                "trainer",
                self.config["stage_key"],
                self.config["learning_rate"],
                float(model.weight.detach()),
            )
        )

    def fit(self, train_loader, val_loader):
        assert list(train_loader) == ["fit_train"]
        assert list(val_loader) == ["val"]
        stage = self.config["stage_key"]
        value = 2.0 if stage == "A" else 3.0
        self.model.weight.data.fill_(value)
        score = 0.75 if stage == "A" else 1.0
        history = [
            {
                "epoch": 1,
                "learning_rate": self.config["learning_rate"],
                "train_loss": 0.2,
                "train_valid_samples": 4,
                "train_rejected_samples": 0,
                "optimizer_steps": 1,
                "gradient_norm": 1.0,
                "val_loss": 0.1,
                "val_AUROC": score,
                "val_valid_samples": 4,
                "val_rejected_samples": 0,
                "val_AUPRC": 0.8 if stage == "A" else 1.0,
                "val_F1": 0.8 if stage == "A" else 1.0,
                "stage": self.config["stage_name"],
                "train_binary_loss": 0.2,
                "val_binary_loss": 0.1,
                "epoch_time_seconds": 1.0,
                "peak_cuda_memory_bytes": 100,
                "checkpoint_decision": "best",
            }
        ]
        checkpoint = {
            "weight": value,
            "model_alias": "hosp_bonenet",
            "selection_metric": "AUROC",
            "selection_value": score,
            "epoch": 1,
            "resolved_config": self.config,
            "history": history,
        }
        for name in ("best.pt", "last.pt"):
            torch.save(checkpoint, self.directory / name)
        return {
            "best_epoch": 1,
            "best_AUROC": score,
            "history": history,
        }

    def predict(self, loader):
        assert list(loader) == ["val"]
        value = float(self.model.weight.detach())
        logits = (
            torch.tensor([-2.0, 1.0, 0.0, 2.0])
            if value == 2.0
            else torch.tensor([-2.0, -1.0, 1.0, 2.0])
        )
        return {
            "logits": logits,
            "targets": torch.tensor([0, 0, 1, 1]),
            "sample_ids": ["val-a", "val-b", "val-c", "val-d"],
            "valid_count": 4,
            "rejected_count": 0,
        }


def _fake_dependencies(events: list[Any]):
    _FakeDataModule.events = events
    _FakeTrainer.events = events

    def load_checkpoint(path, **kwargs):
        payload = torch.load(path, weights_only=True)
        kwargs["model"].weight.data.fill_(payload["weight"])
        events.append(("load", Path(path).parent.name, float(payload["weight"])))
        return {"epoch": 1}

    def metrics(logits, targets, threshold=0.5):
        from bone_cancer_ai.evaluation.classification import binary_classification_metrics

        return binary_classification_metrics(logits, targets, threshold)

    def plot_history(history, destination):
        assert history
        Path(destination).write_bytes(b"png")

    return hosp_cli.RunDependencies(
        data_module_type=_FakeDataModule,
        trainer_type=_FakeTrainer,
        create_model=lambda model_config, schema: _FakeModel(),
        class_weight=lambda frame: torch.tensor(1.0),
        load_checkpoint=load_checkpoint,
        metrics=metrics,
        cuda_available=lambda: True,
        plot_history=plot_history,
    )


def _blocking_dependencies(
    events: list[Any], entered: threading.Event, release: threading.Event
) -> Any:
    dependencies = _fake_dependencies(events)

    class BlockingDataModule(_FakeDataModule):
        def setup(self) -> None:
            super().setup()
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release initial training")

    dependencies.data_module_type = BlockingDataModule
    return dependencies


def test_run_is_two_stage_val_only_and_reloads_selected_checkpoint_exactly(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)
    events: list[Any] = []

    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies(events))

    assert [event for event in events if isinstance(event, str)] == [
        "setup",
        "loader:fit_train",
        "loader:val",
    ]
    trainers = [event for event in events if event[0] == "trainer"]
    assert trainers == [
        ("trainer", "A", 0.0001, 1.0),
        ("trainer", "A", 0.0001, 2.0),
        ("trainer", "D", 0.00001, 2.0),
        ("trainer", "D", 0.00001, 3.0),
    ]
    assert (run / "stage_A" / "validation_predictions.csv").is_file()
    assert (run / "stage_D" / "validation_predictions.csv").is_file()
    assert (run / "stage_A" / "training_curves.png").read_bytes() == b"png"
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected_stage"] == "D"
    assert selection["selected_epoch"] == 1
    assert len(selection["checkpoint_sha256"]) == 64
    assert set(selection["stage_checkpoints"]) == {"A", "D"}
    assert all(
        stage["reload_exact"]
        and len(stage["best_sha256"]) == 64
        and len(stage["last_sha256"]) == 64
        for stage in selection["stage_checkpoints"].values()
    )
    assert selection["skipped_stages"] == ["B", "C"]
    assert (run / "best_checkpoint_path.txt").read_text().strip() == "stage_D/best.pt"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"] == {"status": "not_run", "planned_phase": 9}
    assert metrics["validation"]["AUROC"] == pytest.approx(1.0)
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    assert hosp_cli.validate_run_bundle(run)["selected_stage"] == "D"

    stage_a = run / "stage_A" / "best.pt"
    original_stage_a = stage_a.read_bytes()
    with stage_a.open("ab") as checkpoint:
        checkpoint.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        hosp_cli.validate_run_bundle(run)
    stage_a.write_bytes(original_stage_a)

    with (run / "stage_D" / "best.pt").open("ab") as checkpoint:
        checkpoint.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        hosp_cli.validate_run_bundle(run)


def test_run_bundle_validator_rejects_test_artifacts_and_unsafe_checkpoint_paths(
    tmp_path: Path,
) -> None:
    run = hosp_cli.run_hosp_training(_project_config(tmp_path), dependencies=_fake_dependencies([]))
    run.joinpath("test_predictions.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test artifact"):
        hosp_cli.validate_run_bundle(run)
    run.joinpath("test_predictions.csv").unlink()
    run.joinpath("best_checkpoint_path.txt").write_text("../best.pt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint path"):
        hosp_cli.validate_run_bundle(run)


def test_recorded_input_hashes_detect_post_resolution_manifest_change(tmp_path: Path) -> None:
    resolved = hosp_cli.resolve_config(_project_config(tmp_path))
    runtime = resolved["runtime"]
    val_path = Path(runtime["split_dir"]) / "val.csv"
    val_path.write_text(val_path.read_text(encoding="utf-8") + "changed,unit-generation\n")

    with pytest.raises(ValueError, match="input hash"):
        hosp_cli.validate_recorded_inputs(resolved["record"], runtime)


def test_production_git_guard_rejects_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["a" * 40 + "\n", " M changed.py\n"])

    def fake_run(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(hosp_cli.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="clean committed Git tree"):
        hosp_cli._require_clean_git(ROOT)


def test_failed_run_resumes_stage_a_last_and_continues_missing_stage_d(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    shutil.rmtree(run / "stage_D")
    for name in ("selection.json", "metrics.json", "best_checkpoint_path.txt"):
        (run / name).unlink()
    (run / "status.json").write_text(
        json.dumps({"status": "failed", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )
    events: list[Any] = []

    resumed = hosp_cli.run_hosp_training(
        config,
        dependencies=_fake_dependencies(events),
        resume_run=run,
    )

    assert resumed == run
    assert ("resume", "A", "last.pt") in events
    assert not any(event[:2] == ("resume", "D") for event in events if isinstance(event, tuple))
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    assert (run / "resume_provenance.txt").is_file()
    assert hosp_cli.validate_run_bundle(run)["selected_stage"] == "D"


def test_exact_completed_run_is_validated_registered_and_reused_without_clean_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bone_cancer_ai.artifacts.registry import resolve_checkpoint

    config = _project_config(tmp_path)
    project_root = config.parents[1]
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    registry_path = project_root / "artifacts" / "checkpoints" / "registry.json"
    registry_before = registry_path.read_bytes()

    def reject_git_check(project_root: Path) -> str:
        del project_root
        raise AssertionError("completed-run reuse must not inspect Git cleanliness")

    monkeypatch.setattr(hosp_cli, "_require_clean_git", reject_git_check)
    reused = hosp_cli.run_hosp_training(config)

    assert reused == run
    assert registry_path.read_bytes() == registry_before
    record = resolve_checkpoint(project_root, "hosp_bonenet.phase7.selected")
    assert record.source_run == run.relative_to(project_root).as_posix()
    assert record.selected_metric == {"split": "val", "name": "AUROC", "value": 1.0}


def test_exact_completed_run_must_pass_bundle_validation_before_reuse(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    with (run / "stage_D" / "best.pt").open("ab") as checkpoint:
        checkpoint.write(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))


def test_exact_failed_run_is_automatically_resumed_before_allocating_a_new_run(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    shutil.rmtree(run / "stage_D")
    for name in ("selection.json", "metrics.json", "best_checkpoint_path.txt"):
        (run / name).unlink()
    (run / "status.json").write_text(
        json.dumps({"status": "failed", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )
    events: list[Any] = []

    resumed = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies(events))

    assert resumed == run
    assert ("resume", "A", "last.pt") in events
    assert (run / "resume_provenance.txt").is_file()
    assert json.loads((run / "status.json").read_text())["status"] == "complete"


def test_force_new_run_retrains_and_replaces_the_selected_alias(tmp_path: Path) -> None:
    from bone_cancer_ai.artifacts.registry import resolve_checkpoint

    config = _project_config(tmp_path)
    project_root = config.parents[1]
    first = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))

    second = hosp_cli.run_hosp_training(
        config,
        dependencies=_fake_dependencies([]),
        force_new_run=True,
    )

    assert second != first
    assert first.is_dir() and second.is_dir()
    selected = resolve_checkpoint(project_root, "hosp_bonenet.phase7.selected")
    assert selected.source_run == second.relative_to(project_root).as_posix()


def test_force_new_run_conflicts_with_explicit_resume(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))

    with pytest.raises(ValueError, match="force_new_run.*resume_run"):
        hosp_cli.run_hosp_training(
            config,
            dependencies=_fake_dependencies([]),
            resume_run=run,
            force_new_run=True,
        )


def test_discovery_requires_strict_persisted_config_equality(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    resolved = hosp_cli.resolve_config(config)
    record = resolved["record"]
    run = Path(resolved["runtime"]["output_root"]) / "candidate"
    run.mkdir(parents=True)
    saved_record = dict(record)
    saved_record["seed"] = 42.0
    (run / "config.yaml").write_text(
        yaml.safe_dump(saved_record, sort_keys=False), encoding="utf-8"
    )
    (run / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    assert hosp_cli._matching_run_status(run, record) is None


def test_running_candidate_does_not_shadow_an_exact_failed_resume(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    failed = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    shutil.rmtree(failed / "stage_D")
    for name in ("selection.json", "metrics.json", "best_checkpoint_path.txt"):
        (failed / name).unlink()
    (failed / "status.json").write_text(
        json.dumps({"status": "failed", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )
    resolved = hosp_cli.resolve_config(config)
    running = Path(resolved["runtime"]["output_root"]) / "running-candidate"
    (running / "stage_A").mkdir(parents=True)
    (running / "config.yaml").write_text(
        yaml.safe_dump(resolved["record"], sort_keys=False), encoding="utf-8"
    )
    (running / "status.json").write_text(
        json.dumps({"status": "running", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )
    (running / "stage_A" / "last.pt").write_bytes(b"active checkpoint")
    running_before = {
        path.relative_to(running).as_posix(): path.read_bytes()
        for path in running.rglob("*")
        if path.is_file()
    }

    resumed = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))

    assert resumed == failed
    assert {
        path.relative_to(running).as_posix(): path.read_bytes()
        for path in running.rglob("*")
        if path.is_file()
    } == running_before


def test_running_only_candidate_is_ignored_when_allocating_a_fresh_run(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    resolved = hosp_cli.resolve_config(config)
    record = resolved["record"]
    output_root = Path(resolved["runtime"]["output_root"])
    run = output_root / "candidate"
    (run / "stage_A").mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False), encoding="utf-8"
    )
    (run / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (run / "stage_A" / "last.pt").write_bytes(b"active checkpoint")

    fresh = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))

    assert fresh != run
    assert fresh.parent == output_root
    assert json.loads((run / "status.json").read_text(encoding="utf-8")) == {
        "status": "running"
    }


def test_explicit_resume_reclaims_a_stale_running_run_when_its_lock_is_free(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    shutil.rmtree(run / "stage_D")
    for name in ("selection.json", "metrics.json", "best_checkpoint_path.txt"):
        (run / name).unlink()
    (run / "status.json").write_text(
        json.dumps({"status": "running", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )

    resumed = hosp_cli.run_hosp_training(
        config,
        dependencies=_fake_dependencies([]),
        resume_run=run,
    )

    assert resumed == run
    assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == (
        "complete"
    )
    assert (run / "resume_provenance.txt").is_file()


def test_cli_rejects_force_new_run_with_explicit_resume(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as error:
        hosp_cli.main(["--resume-run", str(tmp_path), "--force-new-run"])

    assert error.value.code == 2
    assert "--force-new-run cannot be combined with --resume-run" in capsys.readouterr().err


@pytest.mark.parametrize("explicit_resume", [False, True])
def test_resume_lock_contention_fails_without_mutating_run_artifacts(
    tmp_path: Path, explicit_resume: bool
) -> None:
    from bone_cancer_ai.utils.locking import exclusive_file_lock

    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    shutil.rmtree(run / "stage_D")
    for name in ("selection.json", "metrics.json", "best_checkpoint_path.txt"):
        (run / name).unlink()
    (run / "status.json").write_text(
        json.dumps({"status": "failed", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )
    acquired = threading.Event()
    release = threading.Event()
    holder_errors: list[BaseException] = []

    def hold_run_lock() -> None:
        try:
            with exclusive_file_lock(run / ".run.lock"):
                acquired.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release the run lock")
        except BaseException as error:
            holder_errors.append(error)

    holder = threading.Thread(target=hold_run_lock)
    holder.start()
    assert acquired.wait(timeout=5)
    before = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file() and path.name != ".run.lock"
    }
    try:
        with pytest.raises(TimeoutError, match="already locked"):
            hosp_cli.run_hosp_training(
                config,
                dependencies=_fake_dependencies([]),
                resume_run=run if explicit_resume else None,
            )
    finally:
        release.set()
        holder.join(timeout=5)

    after = {
        path.relative_to(run).as_posix(): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file() and path.name != ".run.lock"
    }
    assert not holder.is_alive()
    assert holder_errors == []
    assert after == before


@pytest.mark.parametrize("force_new_run", [False, True])
def test_initial_execution_holds_run_lock_against_explicit_resume(
    tmp_path: Path, force_new_run: bool
) -> None:
    config = _project_config(tmp_path)
    if force_new_run:
        hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    entered = threading.Event()
    release = threading.Event()
    initial_errors: list[BaseException] = []

    def run_initial() -> None:
        try:
            hosp_cli.run_hosp_training(
                config,
                dependencies=_blocking_dependencies([], entered, release),
                force_new_run=force_new_run,
            )
        except BaseException as error:
            initial_errors.append(error)

    initial = threading.Thread(target=run_initial)
    initial.start()
    assert entered.wait(timeout=5)
    output_root = config.parents[1] / "experiments" / "hosp_bonenet"
    running_runs = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and json.loads((path / "status.json").read_text(encoding="utf-8"))["status"]
        == "running"
    ]
    assert len(running_runs) == 1
    running = running_runs[0]
    before = {
        path.relative_to(running).as_posix(): path.read_bytes()
        for path in running.rglob("*")
        if path.is_file() and path.name != ".run.lock"
    }
    contender_error: BaseException | None = None
    try:
        try:
            hosp_cli.run_hosp_training(
                config,
                dependencies=_fake_dependencies([]),
                resume_run=running,
            )
        except BaseException as error:
            contender_error = error
        after = {
            path.relative_to(running).as_posix(): path.read_bytes()
            for path in running.rglob("*")
            if path.is_file() and path.name != ".run.lock"
        }
    finally:
        release.set()
        initial.join(timeout=5)

    assert isinstance(contender_error, TimeoutError)
    assert "already locked" in str(contender_error)
    assert after == before
    assert not initial.is_alive()
    assert initial_errors == []


def test_failed_run_with_last_only_checkpoint_is_automatically_recoverable(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    shutil.rmtree(run / "stage_D")
    (run / "stage_A" / "best.pt").unlink()
    for name in ("selection.json", "metrics.json", "best_checkpoint_path.txt"):
        (run / name).unlink()
    (run / "status.json").write_text(
        json.dumps({"status": "failed", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )
    events: list[Any] = []

    resumed = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies(events))

    assert resumed == run
    assert ("resume", "A", "last.pt") in events
    assert (run / "stage_A" / "best.pt").is_file()
    assert hosp_cli.validate_run_bundle(run)["selected_stage"] == "D"


def test_resume_still_rejects_best_without_last(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    run = hosp_cli.run_hosp_training(config, dependencies=_fake_dependencies([]))
    (run / "stage_A" / "last.pt").unlink()
    (run / "status.json").write_text(
        json.dumps({"status": "failed", "test": "not_run", "planned_phase": 9}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete checkpoint pair"):
        hosp_cli.run_hosp_training(
            config,
            dependencies=_fake_dependencies([]),
            resume_run=run,
        )
