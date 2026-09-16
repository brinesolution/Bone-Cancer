"""Phase 8 data screening, tuning, ablation, and publication workflow."""

from __future__ import annotations

import gc
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
    METRIC_TOLERANCE,
    ablation_model_config,
    atomic_text,
    load_completed_stage,
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
    training_stage_definition,
)
from bone_cancer_ai.optimization.phase8 import (
    phase8_fingerprint,
    select_validation_candidate,
)
from bone_cancer_ai.training.trainer import BinaryTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _phase8_source_paths(root: Path) -> dict[str, Path]:
    """Return every package source that can affect Phase 8 execution."""
    root = Path(root).resolve()
    package = root / "src" / "bone_cancer_ai"
    paths = sorted(path for path in package.rglob("*.py") if path.is_file())
    paths.append(root / "scripts" / "11_tune_hosp_bonenet.py")
    paths.append(root / "requirements" / "lock_windows_gpu.txt")
    return {
        f"source:{path.relative_to(root).as_posix()}": path
        for path in sorted(paths)
    }


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


def _remove_pruned_checkpoints(trial_dir: Path) -> None:
    """Discard terminal pruned weights while retaining their study evidence."""
    for name in ("best.pt", "last.pt"):
        (Path(trial_dir) / name).unlink(missing_ok=True)


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
            _remove_pruned_checkpoints(trial_dir)
            raise
        trial.set_user_attr("stage_id", signature)
        trial.set_user_attr("stage_signature", metrics["signature"])
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
        **_phase8_source_paths(root),
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


def _validated_training_stage(
    stage_dir: Path,
    expected_definition: Mapping[str, Any],
) -> dict[str, Any]:
    expected = strict_json(expected_definition, name="expected training stage")
    recorded = load_yaml(stage_dir / "config.yaml")
    if recorded != expected:
        raise ValueError(f"training stage definition differs: {stage_dir}")
    signature = phase8_fingerprint(expected)
    validated = load_completed_stage(stage_dir, signature)
    if validated is None:
        raise ValueError(f"training stage is missing: {stage_dir}")
    if validated.get("architecture") != expected["architecture"]:
        raise ValueError(f"training stage architecture differs: {stage_dir}")
    return validated


def _metric_matches(left: object, right: object) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=METRIC_TOLERANCE,
        )
    except (TypeError, ValueError):
        return False


def _validate_ablation_table(
    context: Mapping[str, Any],
    run_dir: Path,
    ablation: pd.DataFrame,
    selection: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    hparams: Mapping[str, Any],
) -> str:
    """Bind every published ablation row to validated stage artifacts."""
    expected_ids = [str(row["id"]) for row in context["config"]["ablations"]]
    if ablation["ablation"].tolist() != [*expected_ids, "FULL"]:
        raise ValueError("Phase 8 ablation rows are incomplete")
    if not ablation["test_status"].eq("not_run").all():
        raise ValueError("Phase 8 ablation table has invalid test status")
    components = {
        str(row["id"]): str(row["component"])
        for row in context["config"]["ablations"]
    }
    rows = {str(row.ablation): row for row in ablation.itertuples(index=False)}
    stage_config = context["config"]["ablation_run"]
    candidates = []
    for ablation_id in expected_ids:
        row = rows[ablation_id]
        definition = training_stage_definition(
            context,
            architecture=ablation_id,
            settings=settings,
            stage={
                "name": f"ablation_{ablation_id}",
                "epochs": int(stage_config["epochs"]),
                "early_stopping_patience": int(stage_config["early_stopping_patience"]),
                "train_samples": stage_config["train_samples"],
                "val_samples": stage_config["val_samples"],
            },
            hparams=hparams,
        )
        stage = _validated_training_stage(
            run_dir / "ablations" / ablation_id,
            definition,
        )
        expected_text = {
            "component": components[ablation_id],
            "protocol": "phase7_warm_start_branch_ablation",
            "initialization": context["parent_record"].alias,
            "checkpoint_sha256": stage["checkpoint_sha256"],
        }
        for field, expected in expected_text.items():
            if getattr(row, field) != expected:
                raise ValueError(f"Phase 8 {ablation_id} ablation {field} differs")
        for name in METRIC_COLUMNS:
            if not _metric_matches(getattr(row, name), stage["validation"][name]):
                raise ValueError(f"Phase 8 {ablation_id} ablation metric differs: {name}")
        candidates.append(
            {"candidate": ablation_id, "validation": stage["validation"]}
        )

    full = rows["FULL"]
    if (
        full.component != "selected supported binary configuration"
        or full.protocol != selection.get("candidate")
        or full.checkpoint_sha256 != selection.get("checkpoint_sha256")
    ):
        raise ValueError("Phase 8 FULL ablation metadata differs from selection")
    for name in METRIC_COLUMNS:
        if not _metric_matches(getattr(full, name), selection["validation"][name]):
            raise ValueError(f"Phase 8 FULL ablation metric differs: {name}")
    return str(select_validation_candidate(candidates)["candidate"])


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
        "selected_validation_predictions.csv", "provenance.json", "study.db",
        "tuning_selection.json", "data_selection.json",
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
    data_selection = json.loads(
        (run_dir / "data_selection.json").read_text(encoding="utf-8")
    )
    settings = data_selection.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Phase 8 selected data settings are missing")

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
            rel_tol=0.0,
            abs_tol=METRIC_TOLERANCE,
        ):
            raise ValueError(f"selected validation metric differs: {name}")

    trials = pd.read_csv(run_dir / "trials.csv")
    if len(trials) != 10 or trials["trial"].nunique() != 10 or int(trials["selected"].sum()) != 1:
        raise ValueError("Phase 8 trial table is incomplete or ambiguous")

    import optuna

    study = optuna.load_study(
        study_name=f"phase8-{run_dir.name}",
        storage=f"sqlite:///{(run_dir / 'study.db').resolve().as_posix()}",
    )
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    terminal = [trial for trial in study.trials if trial.state in terminal_states]
    if len(terminal) != 10:
        raise ValueError("Phase 8 study does not contain ten terminal trials")
    csv_by_number = {
        int(str(row.trial).split("_")[1]): row
        for row in trials.itertuples(index=False)
    }
    if set(csv_by_number) != {trial.number for trial in terminal}:
        raise ValueError("Phase 8 study trial numbers differ from trials.csv")
    validated_candidates = []
    optimization = context["config"]["optimization"]
    for trial in terminal:
        row = csv_by_number[trial.number]
        if row.state != trial.state.name:
            raise ValueError("Phase 8 study trial state differs from trials.csv")
        for name, value in trial.params.items():
            recorded = getattr(row, name)
            if isinstance(value, float):
                if not math.isclose(float(recorded), value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Phase 8 study parameters differ from trials.csv")
            elif recorded != value:
                raise ValueError("Phase 8 study parameters differ from trials.csv")
        selected_flag = bool(row.selected)
        if trial.state == optuna.trial.TrialState.PRUNED:
            if selected_flag:
                raise ValueError("a pruned Phase 8 trial cannot be selected")
            continue
        validation = trial.user_attrs.get("validation")
        stage_relative = trial.user_attrs.get("stage")
        stage_id = trial.user_attrs.get("stage_id")
        stage_signature = trial.user_attrs.get("stage_signature")
        if (
            not isinstance(validation, dict)
            or not isinstance(stage_relative, str)
            or not isinstance(stage_id, str)
            or not isinstance(stage_signature, str)
        ):
            raise ValueError("completed Phase 8 trial attributes are incomplete")
        hparams = dict(trial.params)
        hparams["gradient_accumulation_steps"] = (
            4 if settings["batch_size"] == 2 else 2
        )
        expected_stage_id = phase8_fingerprint(hparams)[:16]
        expected_relative = (Path("trials") / expected_stage_id).as_posix()
        if stage_id != expected_stage_id or stage_relative != expected_relative:
            raise ValueError("completed Phase 8 trial stage identity differs")
        stage_path = (run_dir / stage_relative).resolve()
        if not stage_path.is_relative_to((run_dir / "trials").resolve()):
            raise ValueError("completed Phase 8 trial stage escapes its run")
        definition = training_stage_definition(
            context,
            architecture="B2",
            settings=settings,
            stage={
                "name": f"optuna_{expected_stage_id}",
                "epochs": int(optimization["epochs"]),
                "early_stopping_patience": int(
                    optimization["early_stopping_patience"]
                ),
                "train_samples": int(optimization["train_samples"]),
                "val_samples": int(optimization["val_samples"]),
            },
            hparams=hparams,
        )
        expected_stage_signature = phase8_fingerprint(definition)
        if stage_signature != expected_stage_signature:
            raise ValueError("completed Phase 8 trial signature differs")
        stage = _validated_training_stage(stage_path, definition)
        if stage.get("validation") != validation:
            raise ValueError("completed Phase 8 trial differs from its stage")
        if not _metric_matches(trial.value, validation["AUROC"]):
            raise ValueError("Phase 8 trial objective differs from validation AUROC")
        for name in METRIC_COLUMNS:
            if not _metric_matches(getattr(row, name), validation[name]):
                raise ValueError(f"Phase 8 trial metric differs from trials.csv: {name}")
        validated_candidates.append(
            {
                "candidate": f"trial_{trial.number:03d}",
                "validation": validation,
                "params": trial.params,
                "stage": stage_relative,
            }
        )

    tuning_selection = json.loads(
        (run_dir / "tuning_selection.json").read_text(encoding="utf-8")
    )
    expected_tuning_selection = select_validation_candidate(validated_candidates)
    selected_rows = [row for row in csv_by_number.values() if bool(row.selected)]
    if len(selected_rows) != 1:
        raise ValueError("Phase 8 tuning selection is ambiguous")
    selected_row = selected_rows[0]
    selected_number = int(str(selected_row.trial).split("_")[1])
    selected_trial = next(trial for trial in terminal if trial.number == selected_number)
    if (
        tuning_selection != expected_tuning_selection
        or tuning_selection.get("candidate") != selected_row.trial
        or tuning_selection.get("params") != selected_trial.params
        or tuning_selection.get("stage") != selected_trial.user_attrs.get("stage")
        or tuning_selection.get("validation") != selected_trial.user_attrs.get("validation")
    ):
        raise ValueError("Phase 8 tuning selection differs from the study")

    best_hparams = dict(tuning_selection["params"])
    best_hparams["gradient_accumulation_steps"] = (
        4 if settings["batch_size"] == 2 else 2
    )
    ablation = pd.read_csv(run_dir / "hosp_ablation.csv")
    best_architecture = _validate_ablation_table(
        context,
        run_dir,
        ablation,
        selection,
        settings=settings,
        hparams=best_hparams,
    )
    final_config = context["config"]["final_run"]
    final_definition = training_stage_definition(
        context,
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
    final_stage = _validated_training_stage(
        run_dir / "final_candidate",
        final_definition,
    )
    incumbent = evaluation_stage(context, run_dir / "incumbent", base_settings())
    expected_selection = select_validation_candidate(
        [
            {
                "candidate": "phase7_incumbent",
                "validation": incumbent["validation"],
            },
            {
                "candidate": "phase8_optimized",
                "validation": final_stage["validation"],
            },
        ]
    )
    if selection.get("candidate") != expected_selection["candidate"]:
        raise ValueError("Phase 8 selection differs from validated candidates")
    selected_stage = (
        incumbent if selection["candidate"] == "phase7_incumbent" else final_stage
    )
    if selection.get("validation") != selected_stage.get("validation"):
        raise ValueError("Phase 8 selection metrics differ from selected candidate")
    if selection.get("checkpoint_sha256") != selected_stage.get("checkpoint_sha256"):
        raise ValueError("Phase 8 selection checkpoint differs from selected candidate")

    module, _ = make_data(context, frozen["data"], train_limit=None, val_limit=None)
    verifier = BinaryTrainer(
        model,
        model_alias="hosp_bonenet_phase8_verifier",
        resolved_config={
            "epochs": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
            "amp": bool(context["amp"]),
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 1,
            "seed": int(context["seed"]),
            "selection_metric": "AUROC",
        },
        checkpoint_dir=run_dir / "verification_only",
        pos_weight=1.0,
        device=context["device"],
    )
    actual = verifier.predict(module.loader("val"))
    if actual["sample_ids"] != predictions["sample_id"].astype(str).tolist():
        raise ValueError("selected checkpoint sample IDs differ from saved predictions")
    saved_targets = torch.as_tensor(
        predictions["target"].to_numpy(copy=True), dtype=torch.long
    )
    saved_logits = torch.as_tensor(
        predictions["logit"].to_numpy(copy=True), dtype=torch.float32
    )
    if not torch.equal(actual["targets"], saved_targets) or not torch.allclose(
        actual["logits"], saved_logits, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("selected checkpoint inference differs from saved predictions")
    del actual, verifier, model, module
    gc.collect()
    if context["device"].type == "cuda":
        torch.cuda.empty_cache()
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
    with checkpoint_publication_lock(root, "hosp_bonenet.phase8.publication"):
        snapshots = {
            global_config: (
                global_config.read_text(encoding="utf-8")
                if global_config.exists()
                else None
            ),
            global_metrics: (
                global_metrics.read_text(encoding="utf-8")
                if global_metrics.exists()
                else None
            ),
        }
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
            rollback_error: Exception | None = None
            try:
                restore_checkpoint_alias(
                    root,
                    alias=alias,
                    expected_current=promotion.promoted,
                    previous=promotion.previous,
                )
            except Exception as error:
                rollback_error = error
            for path, prior in snapshots.items():
                _restore_text(path, prior)
            if rollback_error is not None:
                raise RuntimeError("Phase 8 alias rollback failed") from rollback_error
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
    incumbent_settings = base_settings()
    incumbent = evaluation_stage(context, run_dir / "incumbent", incumbent_settings)
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
    lock_alias = f"hosp_bonenet.phase8.run.f{fingerprint[:12]}"
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
