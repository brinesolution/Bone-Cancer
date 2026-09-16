"""Phase 8 data screening, tuning, ablation, and publication workflow."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from bone_cancer_ai.artifacts.registry import (
    checkpoint_publication_lock,
    promote_checkpoint_transaction,
    resolve_checkpoint,
    restore_checkpoint_alias,
)
from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
from bone_cancer_ai.optimization.contracts import (
    METRIC_COLUMNS,
    ablation_model_config,
    atomic_text,
    load_yaml,
    resolve_phase8_config,
    sha256_file,
    strict_json,
    write_json,
    write_yaml,
)
from bone_cancer_ai.optimization.execution import (
    base_hparams,
    base_settings,
    evaluation_stage,
    make_data,
    seed_everything,
    training_stage,
)
from bone_cancer_ai.optimization.phase8 import (
    phase8_fingerprint,
    select_validation_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _cache_throughput(context: Mapping[str, Any], run_dir: Path) -> None:
    """Benchmark the verified 384 cache against source decoding on the same samples."""
    destination = run_dir / "cache_throughput.csv"
    if destination.is_file():
        return
    batches = int(context["config"]["data_screen"]["benchmark_batches"])
    batch_size = 4
    sample_limit = min(len(context["full_fit_frame"]), batches * batch_size)
    rows = []
    for cache_enabled in (True, False):
        settings = dict(base_settings(), cache_enabled=cache_enabled)
        module, _ = make_data(
            context, settings, train_limit=sample_limit, val_limit=2
        )
        loader = module.loader("fit_train")
        started = time.perf_counter()
        samples = 0
        observed_batches = 0
        for observed_batches, batch in enumerate(loader, start=1):
            samples += int(batch["image"].shape[0])
            if observed_batches >= batches:
                break
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "cache_enabled": cache_enabled,
                "image_size": 384,
                "batches": observed_batches,
                "samples": samples,
                "seconds": elapsed,
                "samples_per_second": samples / elapsed,
            }
        )
    pd.DataFrame(rows).to_csv(destination, index=False)


def _data_screen(context: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    _cache_throughput(context, run_dir)
    screen = context["config"]["data_screen"]
    stage_base = {
        "epochs": int(screen["epochs"]),
        "early_stopping_patience": int(screen["early_stopping_patience"]),
        "train_samples": int(screen["train_samples"]),
        "val_samples": int(screen["val_samples"]),
    }
    hparams = base_hparams()
    rows: list[dict[str, Any]] = []
    balance_results: list[dict[str, Any]] = []
    for strategy in screen["balance_strategies"]:
        settings = base_settings()
        settings["balance_strategy"] = strategy
        metrics = training_stage(
            context,
            run_dir / "data_screen" / f"balance_{strategy}",
            architecture="B2",
            settings=settings,
            stage=dict(stage_base, name=f"data_balance_{strategy}"),
            hparams=hparams,
        )
        row = {
            "candidate": strategy,
            "category": "balance",
            "settings": settings,
            "validation": metrics["validation"],
        }
        rows.append(row)
        balance_results.append(row)
    selected_balance = select_validation_candidate(balance_results)
    augmentation_results = [
        {
            "candidate": "current",
            "category": "augmentation",
            "settings": selected_balance["settings"],
            "validation": selected_balance["validation"],
        }
    ]
    if "weak" in screen["augmentation_profiles"]:
        settings = dict(selected_balance["settings"], augmentation_profile="weak")
        metrics = training_stage(
            context,
            run_dir / "data_screen" / "augmentation_weak",
            architecture="B2",
            settings=settings,
            stage=dict(stage_base, name="data_augmentation_weak"),
            hparams=hparams,
        )
        augmentation_results.append(
            {
                "candidate": "weak",
                "category": "augmentation",
                "settings": settings,
                "validation": metrics["validation"],
            }
        )
    rows.extend(augmentation_results)
    selected_augmentation = select_validation_candidate(augmentation_results)
    size_results = [
        {
            "candidate": "384",
            "category": "image_size",
            "settings": selected_augmentation["settings"],
            "validation": selected_augmentation["validation"],
        }
    ]
    if 512 in screen["image_sizes"]:
        settings = dict(
            selected_augmentation["settings"],
            image_size=512,
            batch_size=2,
            cache_enabled=False,
        )
        try:
            metrics = training_stage(
                context,
                run_dir / "data_screen" / "image_size_512",
                architecture="B2",
                settings=settings,
                stage=dict(stage_base, name="data_image_size_512"),
                hparams=dict(hparams, gradient_accumulation_steps=4),
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            metrics = None
        if metrics is not None:
            size_results.append(
                {
                    "candidate": "512",
                    "category": "image_size",
                    "settings": settings,
                    "validation": metrics["validation"],
                }
            )
    rows.extend(size_results)
    selected_size = select_validation_candidate(size_results)
    flat_rows = []
    for row in rows:
        flat = {"category": row["category"], "candidate": row["candidate"]}
        flat.update({f"setting_{key}": value for key, value in row["settings"].items()})
        flat.update({key: row["validation"].get(key) for key in METRIC_COLUMNS})
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).drop_duplicates().to_csv(run_dir / "data_screen.csv", index=False)
    result = {
        "settings": selected_size["settings"],
        "balance": selected_balance["candidate"],
        "augmentation": selected_augmentation["candidate"],
        "image_size": int(selected_size["candidate"]),
    }
    write_json(run_dir / "data_selection.json", result)
    return result


def _run_optuna(
    context: Mapping[str, Any], run_dir: Path, settings: Mapping[str, Any]
) -> dict[str, Any]:
    import optuna

    config = context["config"]["optimization"]
    search_space = {
        "learning_rate": list(config["learning_rates"]),
        "weight_decay": list(config["weight_decays"]),
        "warmup_epochs": list(config["warmup_epochs"]),
        "dropout": list(config["dropouts"]),
        "label_smoothing": list(config["label_smoothing"]),
    }
    storage_path = (run_dir / "study.db").resolve()
    study = optuna.create_study(
        study_name=f"phase8-{run_dir.name}",
        storage=f"sqlite:///{storage_path.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space, seed=int(context["seed"])),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        load_if_exists=True,
    )
    for stale in [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING
    ]:
        if set(stale.params) == set(search_space):
            study.enqueue_trial(stale.params)
        study._storage.set_trial_state_values(  # noqa: SLF001
            stale._trial_id, optuna.trial.TrialState.FAIL  # noqa: SLF001
        )

    def objective(trial: Any) -> float:
        hparams = {
            key: trial.suggest_categorical(key, values)
            for key, values in search_space.items()
        }
        hparams["gradient_accumulation_steps"] = 4 if settings["batch_size"] == 2 else 2
        signature = phase8_fingerprint(hparams)[:16]

        def report_epoch(record: dict[str, Any]) -> None:
            trial.report(float(record["val_AUROC"]), step=int(record["epoch"]))
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"median pruning at epoch {int(record['epoch'])}"
                )

        trial_dir = run_dir / "trials" / signature
        try:
            metrics = training_stage(
                context,
                trial_dir,
                architecture="B2",
                settings=settings,
                stage={
                    "name": f"optuna_{signature}",
                    "epochs": int(config["epochs"]),
                    "early_stopping_patience": int(config["early_stopping_patience"]),
                    "train_samples": int(config["train_samples"]),
                    "val_samples": int(config["val_samples"]),
                },
                hparams=hparams,
                epoch_callback=report_epoch,
            )
        except optuna.TrialPruned:
            write_json(
                trial_dir / "status.json",
                {"status": "pruned", "trial": trial.number, "signature": signature},
            )
            raise
        trial.set_user_attr("signature", signature)
        trial.set_user_attr("validation", strict_json(metrics["validation"], name="metrics"))
        trial.set_user_attr("stage", (Path("trials") / signature).as_posix())
        return float(metrics["validation"]["AUROC"])

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    terminal = [trial for trial in study.trials if trial.state in terminal_states]
    remaining = int(config["trials"]) - len(terminal)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    terminal = [trial for trial in study.trials if trial.state in terminal_states]
    signatures = {
        phase8_fingerprint(trial.params)
        for trial in terminal
    }
    if len(terminal) != int(config["trials"]) or len(signatures) != len(terminal):
        raise RuntimeError("Optuna study does not contain ten unique terminal trials")
    complete = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError("Optuna study has no complete trial to select")
    candidates = [
        {
            "candidate": f"trial_{trial.number:03d}",
            "validation": trial.user_attrs["validation"],
            "params": trial.params,
            "stage": trial.user_attrs["stage"],
        }
        for trial in complete
    ]
    selected = select_validation_candidate(candidates)
    trial_rows = []
    by_number = {int(candidate["candidate"].split("_")[1]): candidate for candidate in candidates}
    for trial in terminal:
        candidate = by_number.get(trial.number)
        row = {"trial": f"trial_{trial.number:03d}", "state": trial.state.name, **trial.params}
        row.update(
            {
                key: candidate["validation"].get(key) if candidate is not None else None
                for key in METRIC_COLUMNS
            }
        )
        row["selected"] = candidate is not None and candidate["candidate"] == selected["candidate"]
        trial_rows.append(row)
    pd.DataFrame(trial_rows).sort_values("trial").to_csv(run_dir / "trials.csv", index=False)
    write_json(run_dir / "tuning_selection.json", selected)
    return selected


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError("selected checkpoint destination already differs")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _context(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = PROJECT_ROOT
    config_path = config_path.resolve()
    config_dir = (root / "configs").resolve()
    if config_path != config_dir / "optimization.yaml":
        raise ValueError("Phase 8 must use the canonical configs/optimization.yaml")

    def config_reference(value: object, name: str) -> Path:
        if not isinstance(value, str) or Path(value).name != value:
            raise ValueError(f"{name} must name one file in configs")
        resolved = (config_dir / value).resolve()
        if resolved.parent != config_dir:
            raise ValueError(f"{name} escapes the configs directory")
        return resolved

    model_path = config_reference(config["model_config"], "model_config")
    data_path = config_reference(config["data_config"], "data_config")
    schema_path = (root / str(config["schema_path"])).resolve()
    split_dir = (root / str(config["split_dir"])).resolve()
    if schema_path != (root / "workspace" / "annotations" / "label_schema.yaml").resolve():
        raise ValueError("schema_path must be the canonical verified label schema")
    if split_dir != (root / "workspace" / "splits").resolve():
        raise ValueError("split_dir must be the canonical verified split directory")
    output_root = (root / str(config["output_root"])).resolve()
    approved_output = (root / "experiments" / "optimization" / "hosp_bonenet").resolve()
    if output_root != approved_output or output_root.is_relative_to(root / "dataset" / "Main"):
        raise ValueError("output_root must be the approved Phase 8 experiment directory")
    model_document = load_yaml(model_path)
    data_document = load_yaml(data_path)
    schema = load_yaml(schema_path)
    if model_document.get("verified_supervision") != ["binary"]:
        raise ValueError("Phase 8 requires verified binary-only supervision")
    base_model = model_document.get("model")
    if not isinstance(base_model, dict):
        raise ValueError("HOSP model configuration is missing")
    optional = ("subtype_enabled", "risk_enabled", "concepts_enabled")
    if any(bool(base_model.get(key)) for key in optional):
        raise ValueError("optional HOSP supervision must remain disabled")
    parent_record = resolve_checkpoint(root, str(config["parent_alias"]))
    if parent_record.phase != "phase_7" or parent_record.test_status.get("status") != "not_run":
        raise ValueError("Phase 8 parent must be the pre-test Phase 7 checkpoint")
    parent_path = parent_record.absolute_checkpoint_path(root)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("model_alias") != "hosp_bonenet" or not isinstance(
        parent.get("model_state"), Mapping
    ):
        raise ValueError("Phase 7 parent checkpoint is incompatible")
    input_paths = {
        "phase8_config": config_path,
        "model_config": model_path,
        "data_config": data_path,
        "schema": schema_path,
        "fit_train": split_dir / "fit_train.csv",
        "val": split_dir / "val.csv",
        "phase8_contracts": root / "src/bone_cancer_ai/optimization/contracts.py",
        "phase8_execution": root / "src/bone_cancer_ai/optimization/execution.py",
        "phase8_workflow": root / "src/bone_cancer_ai/optimization/workflow.py",
        "hosp_model": root / "src/bone_cancer_ai/models/hosp_bonenet/model.py",
        "binary_trainer": root / "src/bone_cancer_ai/training/trainer.py",
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    full_fit_frame = pd.read_csv(input_paths["fit_train"])
    if not full_fit_frame["assigned_split"].eq("fit_train").all():
        raise ValueError("fit manifest contains non-fit assignments")
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 8 requires CUDA; refusing silent CPU fallback")
    return {
        "root": root,
        "config": config,
        "seed": int(config["seed"]),
        "amp": bool(config["amp"]),
        "device": device,
        "split_dir": split_dir,
        "schema": schema,
        "data_document": data_document,
        "base_model": base_model,
        "parent_record": parent_record,
        "parent_path": parent_path,
        "parent_state": parent["model_state"],
        "input_hashes": input_hashes,
        "full_fit_frame": full_fit_frame,
    }


def _run_directory(
    context: Mapping[str, Any], *, force_new_run: bool
) -> tuple[Path, str]:
    fingerprint = phase8_fingerprint(
        {
            "config": context["config"],
            "inputs": context["input_hashes"],
            "parent_sha256": context["parent_record"].checkpoint_sha256,
        }
    )
    root = (context["root"] / str(context["config"]["output_root"])).resolve()
    if force_new_run:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return root / f"{stamp}-{fingerprint[:12]}", fingerprint
    return root / f"phase8-{fingerprint[:12]}", fingerprint


def _validate_local_bundle(
    context: Mapping[str, Any],
    run_dir: Path,
    fingerprint: str,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("fingerprint") != fingerprint:
        raise ValueError("Phase 8 run fingerprint is invalid")
    if require_complete and status.get("status") != "complete":
        raise ValueError("Phase 8 run is not complete")
    required = {
        "final_config.yaml", "selection.json", "hosp_ablation.csv", "trials.csv",
        "data_screen.csv", "cache_throughput.csv", "selected.pt",
        "selected_validation_predictions.csv", "provenance.json",
    }
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"completed Phase 8 run is missing files: {missing}")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "selected.pt") != selection.get("checkpoint_sha256"):
        raise ValueError("completed Phase 8 selected checkpoint checksum differs")
    frozen = load_yaml(run_dir / "final_config.yaml")
    if phase8_fingerprint(frozen) != selection.get("configuration_fingerprint"):
        raise ValueError("frozen configuration fingerprint differs from selection")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("inputs") != context["input_hashes"]:
        raise ValueError("Phase 8 provenance input hashes are stale")

    model = create_hosp_bonenet(frozen["model"], context["schema"])
    checkpoint = torch.load(run_dir / "selected.pt", map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("selected checkpoint model state is missing")
    model.load_state_dict(state, strict=True)

    predictions = pd.read_csv(run_dir / "selected_validation_predictions.csv")
    if predictions.empty or set(predictions) != {"sample_id", "target", "logit", "probability"}:
        raise ValueError("selected validation predictions are malformed")
    recomputed = binary_classification_metrics(predictions["logit"], predictions["target"])
    for name in METRIC_COLUMNS:
        if not math.isclose(
            float(recomputed[name]),
            float(selection["validation"][name]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"selected validation metric differs: {name}")

    ablation = pd.read_csv(run_dir / "hosp_ablation.csv")
    if ablation["ablation"].tolist() != ["B0", "B1", "B2", "FULL"]:
        raise ValueError("Phase 8 ablation rows are incomplete")
    if not ablation["test_status"].eq("not_run").all():
        raise ValueError("Phase 8 ablation table has invalid test status")
    trials = pd.read_csv(run_dir / "trials.csv")
    if len(trials) != 10 or trials["trial"].nunique() != 10 or int(trials["selected"].sum()) != 1:
        raise ValueError("Phase 8 trial table is incomplete or ambiguous")
    return selection


def _restore_text(path: Path, prior: str | None) -> None:
    if prior is None:
        path.unlink(missing_ok=True)
    else:
        atomic_text(path, prior)


def _publish_selection(
    context: Mapping[str, Any], run_dir: Path, fingerprint: str
) -> dict[str, Any]:
    selection = _validate_local_bundle(
        context, run_dir, fingerprint, require_complete=False
    )
    root = context["root"]
    alias = "hosp_bonenet.phase8.selected"
    global_config = root / "configs" / "hosp_bonenet_phase8_final.yaml"
    global_metrics = root / "artifacts" / "metrics" / "hosp_ablation.csv"
    snapshots = {
        global_config: (
            global_config.read_text(encoding="utf-8") if global_config.exists() else None
        ),
        global_metrics: (
            global_metrics.read_text(encoding="utf-8") if global_metrics.exists() else None
        ),
    }
    with checkpoint_publication_lock(root, "hosp_bonenet.phase8.publication"):
        promotion = promote_checkpoint_transaction(
            root,
            alias=alias,
            checkpoint_path=run_dir / "selected.pt",
            replace_alias=True,
            model_family="hosp_bonenet",
            phase="phase_8",
            source_run=run_dir,
            selected_metric={
                "split": "val",
                "name": "AUROC",
                "value": float(selection["validation"]["AUROC"]),
            },
            configuration_fingerprint=selection["configuration_fingerprint"],
            test_status={"status": "not_run", "planned_phase": 9},
        )
        try:
            atomic_text(
                global_config,
                (run_dir / "final_config.yaml").read_text(encoding="utf-8"),
            )
            atomic_text(
                global_metrics,
                (run_dir / "hosp_ablation.csv").read_text(encoding="utf-8"),
            )
            if resolve_checkpoint(root, alias) != promotion.promoted:
                raise ValueError("Phase 8 alias did not resolve after promotion")
            write_json(
                run_dir / "status.json",
                {"status": "complete", "fingerprint": fingerprint, "phase": 8},
            )
        except Exception:
            restore_checkpoint_alias(
                root,
                alias=alias,
                expected_current=promotion.promoted,
                previous=promotion.previous,
            )
            for path, prior in snapshots.items():
                _restore_text(path, prior)
            raise
    return selection


def _run_phase8_locked(
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            _validate_local_bundle(context, run_dir, fingerprint, require_complete=True)
            result = _publish_selection(context, run_dir, fingerprint)
            print(f"phase8_reuse run={run_dir}", flush=True)
            return result
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"status": "running", "fingerprint": fingerprint, "phase": 8})
    write_yaml(run_dir / "resolved_config.yaml", config)
    write_json(
        run_dir / "provenance.json",
        {
            "inputs": context["input_hashes"],
            "parent": context["parent_record"].as_dict(),
            "allowed_splits": ["fit_train", "val"],
        },
    )
    seed_everything(int(context["seed"]))
    data_selection = _data_screen(context, run_dir)
    settings = data_selection["settings"]
    tuned = _run_optuna(context, run_dir, settings)
    best_hparams = dict(tuned["params"])
    best_hparams["gradient_accumulation_steps"] = 4 if settings["batch_size"] == 2 else 2
    ablation_config = config["ablation_run"]
    ablation_results: list[dict[str, Any]] = []
    for row in config["ablations"]:
        ablation_id = row["id"]
        metrics = training_stage(
            context,
            run_dir / "ablations" / ablation_id,
            architecture=ablation_id,
            settings=settings,
            stage={
                "name": f"ablation_{ablation_id}",
                "epochs": int(ablation_config["epochs"]),
                "early_stopping_patience": int(ablation_config["early_stopping_patience"]),
                "train_samples": ablation_config["train_samples"],
                "val_samples": ablation_config["val_samples"],
            },
            hparams=best_hparams,
        )
        ablation_results.append(
            {
                "candidate": ablation_id,
                "component": row["component"],
                "validation": metrics["validation"],
                "metrics": metrics,
            }
        )
    best_architecture = select_validation_candidate(ablation_results)["candidate"]
    final_config = config["final_run"]
    final_metrics = training_stage(
        context,
        run_dir / "final_candidate",
        architecture=best_architecture,
        settings=settings,
        stage={
            "name": "final_confirmation",
            "epochs": int(final_config["epochs"]),
            "early_stopping_patience": int(final_config["early_stopping_patience"]),
            "train_samples": final_config["train_samples"],
            "val_samples": final_config["val_samples"],
        },
        hparams=best_hparams,
    )
    incumbent = evaluation_stage(context, run_dir / "incumbent", settings)
    selected = select_validation_candidate(
        [
            {
                "candidate": "phase7_incumbent",
                "architecture": "B2",
                "validation": incumbent["validation"],
                "checkpoint": str(context["parent_path"]),
            },
            {
                "candidate": "phase8_optimized",
                "architecture": best_architecture,
                "validation": final_metrics["validation"],
                "checkpoint": str(run_dir / "final_candidate" / "best.pt"),
            },
        ]
    )
    selected_path = run_dir / "selected.pt"
    _link_or_copy(Path(selected["checkpoint"]), selected_path)
    selected_sha = sha256_file(selected_path)
    selected_hparams = (
        best_hparams if selected["candidate"] == "phase8_optimized" else base_hparams()
    )
    selected_settings = (
        settings if selected["candidate"] == "phase8_optimized" else base_settings()
    )
    frozen = {
        "status": "phase_8_frozen",
        "checkpoint_contract": "inference_manifest_v1",
        "selection_split": "val",
        "selection_metric": "AUROC",
        "candidate": selected["candidate"],
        "architecture": selected["architecture"],
        "model": ablation_model_config(
            context["base_model"],
            selected["architecture"],
            dropout=float(selected_hparams["dropout"]),
        ),
        "training": selected_hparams,
        "data": selected_settings,
        "parent_alias": context["parent_record"].alias,
        "parent_sha256": context["parent_record"].checkpoint_sha256,
        "test_evaluation": "not_run",
    }
    write_yaml(run_dir / "final_config.yaml", frozen)
    table_rows = []
    for result in ablation_results:
        metrics = result["metrics"]
        table_row = {
            "ablation": result["candidate"],
            "component": result["component"],
            "protocol": "phase7_warm_start_branch_ablation",
            "initialization": "hosp_bonenet.phase7.selected",
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "test_status": "not_run",
        }
        table_row.update({key: metrics["validation"].get(key) for key in METRIC_COLUMNS})
        table_rows.append(table_row)
    full_row = {
        "ablation": "FULL",
        "component": "selected supported binary configuration",
        "protocol": selected["candidate"],
        "checkpoint_sha256": selected_sha,
        "test_status": "not_run",
    }
    full_row.update({key: selected["validation"].get(key) for key in METRIC_COLUMNS})
    table_rows.append(full_row)
    ablation_frame = pd.DataFrame(table_rows)
    ablation_frame.to_csv(run_dir / "hosp_ablation.csv", index=False)
    selected_predictions_source = (
        run_dir / "final_candidate" / "validation_predictions.csv"
        if selected["candidate"] == "phase8_optimized"
        else run_dir / "incumbent" / "validation_predictions.csv"
    )
    atomic_text(
        run_dir / "selected_validation_predictions.csv",
        selected_predictions_source.read_text(encoding="utf-8"),
    )
    selection = {
        "status": "complete",
        "candidate": selected["candidate"],
        "architecture": selected["architecture"],
        "validation": selected["validation"],
        "checkpoint": "selected.pt",
        "checkpoint_sha256": selected_sha,
        "alias": "hosp_bonenet.phase8.selected",
        "configuration_fingerprint": phase8_fingerprint(frozen),
        "test_status": "not_run",
        "run_dir": run_dir.relative_to(context["root"]).as_posix(),
    }
    write_json(run_dir / "selection.json", selection)
    _validate_local_bundle(context, run_dir, fingerprint, require_complete=False)
    return _publish_selection(context, run_dir, fingerprint)


def run_phase8(config_path: Path, *, force_new_run: bool = False) -> dict[str, Any]:
    """Execute or resume Phase 8 under one process and publication lock."""
    config_path = Path(config_path).resolve()
    config = resolve_phase8_config(config_path)
    context = _context(config_path, config)
    run_dir, fingerprint = _run_directory(context, force_new_run=force_new_run)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_alias = f"hosp_bonenet.phase8.run.{fingerprint[:12]}"
    with checkpoint_publication_lock(context["root"], lock_alias):
        try:
            return _run_phase8_locked(context, config, run_dir, fingerprint)
        except Exception as error:
            write_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "fingerprint": fingerprint,
                    "phase": 8,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


__all__ = ["PROJECT_ROOT", "run_phase8"]
