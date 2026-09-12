"""Run a two-stage real-data CUDA preflight before full Phase 7 training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "10_train_hosp_bonenet.py"


def _load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("phase7_training", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training module: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _balanced_loader(dataset: Any, *, batch_size: int, seed: int) -> DataLoader:
    labels = pd.to_numeric(dataset.frame["binary_label"], errors="raise").astype(int)
    per_class = batch_size
    indices: list[int] = []
    for label in (0, 1):
        selected = labels.index[labels.eq(label)].tolist()[:per_class]
        if len(selected) != per_class:
            raise ValueError(f"smoke subset lacks {per_class} samples for class {label}")
        indices.extend(selected)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_smoke(config_path: Path, output_path: Path) -> dict[str, Any]:
    from bone_cancer_ai.data.datamodule import BoneCancerDataModule
    from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
    from bone_cancer_ai.training.checkpointing import load_checkpoint
    from bone_cancer_ai.training.optimizer import class_weight_from_frame
    from bone_cancer_ai.training.trainer import BinaryTrainer

    phase7 = _load_training_module()
    resolved = phase7.resolve_config(config_path)
    record, runtime = resolved["record"], resolved["runtime"]
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 smoke requires CUDA")
    phase7._set_reproducibility(int(record["seed"]))
    data = BoneCancerDataModule(
        Path(runtime["split_dir"]),
        runtime["data"],
        runtime["schema"],
        splits=("fit_train", "val"),
    )
    data.setup()
    microbatch = int(record["microbatch_size"])
    train_loader = _balanced_loader(
        data.datasets["fit_train"], batch_size=microbatch, seed=int(record["seed"])
    )
    val_loader = _balanced_loader(
        data.datasets["val"], batch_size=microbatch, seed=int(record["seed"]) + 1
    )
    pos_weight = class_weight_from_frame(data.frames["fit_train"])
    run_dir = PROJECT_ROOT / "logs" / (
        "phase7-smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    stages: dict[str, dict[str, Any]] = {}
    stage_predictions: dict[str, dict[str, Any]] = {}
    stage_configs: dict[str, dict[str, Any]] = {}
    stage_checkpoint_paths: dict[str, Path] = {}
    previous_checkpoint: Path | None = None
    previous_config: dict[str, Any] | None = None

    for stage_key in ("A", "D"):
        stage_dir = run_dir / f"stage_{stage_key}"
        model = create_hosp_bonenet(record["model"], runtime["schema"])
        smoke_config = phase7._stage_config(record, stage_key)
        smoke_config.update(
            {
                "epochs": 1,
                "warmup_epochs": 0,
                "early_stopping_patience": 1,
                "smoke_batches": 2,
            }
        )
        if previous_checkpoint is not None and previous_config is not None:
            load_checkpoint(
                previous_checkpoint,
                model=model,
                expected_model_alias="hosp_bonenet",
                map_location="cuda",
                expected_resolved_config=previous_config,
                expected_selection_metric="AUROC",
            )
        trainer = BinaryTrainer(
            model,
            model_alias="hosp_bonenet",
            resolved_config=smoke_config,
            checkpoint_dir=stage_dir,
            pos_weight=pos_weight,
            device="cuda",
        )
        result = trainer.fit(train_loader, val_loader)
        load_checkpoint(
            stage_dir / "best.pt",
            model=model,
            expected_model_alias="hosp_bonenet",
            map_location="cuda",
            expected_resolved_config=smoke_config,
            expected_selection_metric="AUROC",
        )
        prediction = trainer.predict(val_loader)
        stages[stage_key] = {
            "best_epoch": int(result["best_epoch"]),
            "best_AUROC": float(result["best_AUROC"]),
            "history": result["history"],
            "checkpoint_sha256": _sha256(stage_dir / "best.pt"),
        }
        stage_predictions[stage_key] = prediction
        stage_configs[stage_key] = smoke_config
        stage_checkpoint_paths[stage_key] = stage_dir / "best.pt"
        previous_checkpoint = stage_dir / "best.pt"
        previous_config = smoke_config

    reload_exact_by_stage: dict[str, bool] = {}
    for stage_key in ("A", "D"):
        fresh_model = create_hosp_bonenet(record["model"], runtime["schema"])
        load_checkpoint(
            stage_checkpoint_paths[stage_key],
            model=fresh_model,
            expected_model_alias="hosp_bonenet",
            map_location="cuda",
            expected_resolved_config=stage_configs[stage_key],
            expected_selection_metric="AUROC",
        )
        fresh_trainer = BinaryTrainer(
            fresh_model,
            model_alias="hosp_bonenet",
            resolved_config=stage_configs[stage_key],
            checkpoint_dir=run_dir / f"reload_{stage_key}_unused",
            pos_weight=pos_weight,
            device="cuda",
        )
        reloaded = fresh_trainer.predict(val_loader)
        reload_exact_by_stage[stage_key] = phase7._same_prediction(
            stage_predictions[stage_key], reloaded
        )
    if not all(reload_exact_by_stage.values()):
        raise RuntimeError("a smoke stage checkpoint did not reproduce exact predictions")
    selected_stage = phase7.select_final_stage(stages)
    evidence = {
        "status": "pass",
        "purpose": "two-stage real-data optimizer/checkpoint preflight",
        "run_directory": str(run_dir.resolve()),
        "device": torch.cuda.get_device_name(0),
        "image_size": int(record["image_size"]),
        "microbatch_size": microbatch,
        "gradient_accumulation_steps": int(record["gradient_accumulation_steps"]),
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "splits_constructed": ["fit_train", "val"],
        "stages": stages,
        "stage_d_initialized_from_stage_a_best": True,
        "selected_stage": selected_stage,
        "reload_exact_by_stage": reload_exact_by_stage,
        "selected_reload_exact": reload_exact_by_stage[selected_stage],
        "test_status": "not_run",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "hosp_bonenet.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "verification" / "phase7_real_smoke.json",
    )
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
