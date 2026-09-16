"""Data and training execution for resumable Phase 8 stages."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from bone_cancer_ai.data.datamodule import BoneCancerDataModule
from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
from bone_cancer_ai.optimization.contracts import (
    METRIC_COLUMNS,
    ablation_model_config,
    load_completed_stage,
    sha256_file,
    write_json,
    write_yaml,
)
from bone_cancer_ai.optimization.phase8 import (
    load_compatible_parent_state,
    phase8_fingerprint,
    stratified_subset,
)
from bone_cancer_ai.training.optimizer import class_weight_from_frame
from bone_cancer_ai.training.trainer import BinaryTrainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset_module(
    module: BoneCancerDataModule, split: str, limit: int | None, seed: int
) -> None:
    if limit is None or limit == len(module.frames[split]):
        return
    original = module.frames[split].reset_index(drop=True)
    selected = stratified_subset(original, limit=limit, seed=seed)
    positions = {str(value): index for index, value in enumerate(original["sample_id"])}
    indices = [positions[str(value)] for value in selected["sample_id"]]
    module.frames[split] = selected
    module.datasets[split] = Subset(module.datasets[split], indices)


def _augmentation(profile: str, current: Mapping[str, Any]) -> dict[str, Any]:
    if profile == "current":
        return deepcopy(dict(current))
    if profile == "weak":
        return {
            "horizontal_flip": False,
            "vertical_flip": False,
            "rotation_limit": 4,
            "translate_limit": 0.02,
            "scale_limit": 0.03,
            "brightness_contrast_limit": 0.05,
            "gamma_limit": [95, 105],
            "gaussian_noise_std_range": [0.005, 0.015],
            "clahe": False,
        }
    raise ValueError(f"unknown augmentation profile: {profile}")


def make_data(
    context: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    train_limit: int | None,
    val_limit: int | None,
) -> tuple[BoneCancerDataModule, dict[str, Any]]:
    config = deepcopy(context["data_document"])
    config["image_size"] = int(settings["image_size"])
    config["augmentation"] = _augmentation(
        str(settings["augmentation_profile"]), config.get("augmentation", {})
    )
    config["loader"] = deepcopy(config.get("loader", {}))
    config["loader"].update(
        {
            "batch_size": int(settings["batch_size"]),
            "num_workers": int(settings.get("num_workers", 0)),
            "pin_memory": True,
            "persistent_workers": False,
            "sampling": (
                "weighted" if settings["balance_strategy"] == "weighted_sampler" else "shuffle"
            ),
        }
    )
    config["cache"] = deepcopy(config.get("cache", {}))
    config["cache"]["enabled"] = bool(settings["cache_enabled"])
    module = BoneCancerDataModule(
        context["split_dir"], config, context["schema"], splits=("fit_train", "val")
    )
    module.setup()
    _subset_module(module, "fit_train", train_limit, int(context["seed"]))
    _subset_module(module, "val", val_limit, int(context["seed"]) + 1)
    return module, config


def _trainer_config(
    context: Mapping[str, Any], stage: Mapping[str, Any], hparams: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "epochs": int(stage["epochs"]),
        "learning_rate": float(hparams["learning_rate"]),
        "weight_decay": float(hparams["weight_decay"]),
        "warmup_epochs": int(hparams["warmup_epochs"]),
        "label_smoothing": float(hparams["label_smoothing"]),
        "amp": bool(context["amp"]),
        "gradient_accumulation_steps": int(hparams["gradient_accumulation_steps"]),
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": int(stage["early_stopping_patience"]),
        "seed": int(context["seed"]),
        "selection_metric": "AUROC",
        "monitor_extended": True,
        "stage_name": str(stage["name"]),
    }


def _deduplicate_best_last(
    stage_dir: Path, *, best_epoch: int, last_epoch: int
) -> None:
    best = stage_dir / "best.pt"
    last = stage_dir / "last.pt"
    if not best.is_file() or not last.is_file():
        return
    if best_epoch != last_epoch:
        return
    last.unlink()
    try:
        os.link(best, last)
    except OSError:
        shutil.copy2(best, last)


def training_stage(
    context: Mapping[str, Any],
    stage_dir: Path,
    *,
    architecture: str,
    settings: Mapping[str, Any],
    stage: Mapping[str, Any],
    hparams: Mapping[str, Any],
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    model_config = ablation_model_config(
        context["base_model"], architecture, dropout=float(hparams["dropout"])
    )
    trainer_config = _trainer_config(context, stage, hparams)
    definition = {
        "architecture": architecture,
        "model": model_config,
        "training": trainer_config,
        "data": dict(settings),
        "train_samples": stage["train_samples"],
        "val_samples": stage["val_samples"],
        "parent_sha256": context["parent_record"].checkpoint_sha256,
        "inputs": context["input_hashes"],
    }
    signature = phase8_fingerprint(definition)
    completed = load_completed_stage(stage_dir, signature)
    if completed is not None:
        print(f"phase8_reuse stage={stage_dir.name}", flush=True)
        return completed
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(stage_dir / "config.yaml", definition)
    write_json(stage_dir / "status.json", {"status": "running", "signature": signature})
    module, _ = make_data(
        context,
        settings,
        train_limit=stage["train_samples"],
        val_limit=stage["val_samples"],
    )
    loaders = module.loaders()
    model = create_hosp_bonenet(model_config, context["schema"])
    transfer = load_compatible_parent_state(model, context["parent_state"])
    pos_weight = (
        class_weight_from_frame(context["full_fit_frame"])
        if settings["balance_strategy"] == "weighted_bce"
        else torch.tensor(1.0)
    )
    resume_path = stage_dir / "last.pt"
    trainer = BinaryTrainer(
        model,
        model_alias=f"hosp_bonenet_phase8_{architecture.lower()}",
        resolved_config=trainer_config,
        checkpoint_dir=stage_dir,
        pos_weight=pos_weight,
        device=context["device"],
        resume_from=resume_path if resume_path.is_file() else None,
        epoch_callback=epoch_callback,
    )
    result = trainer.fit(loaders["fit_train"], loaders["val"])
    best_path = Path(result["best_checkpoint"])
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    gc.collect()
    if context["device"].type == "cuda":
        torch.cuda.empty_cache()
    prediction = trainer.predict(loaders["val"])
    validation = binary_classification_metrics(prediction["logits"], prediction["targets"])
    for name in METRIC_COLUMNS:
        value = validation.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"validation metric is undefined: {name}")
    pd.DataFrame(
        {
            "sample_id": prediction["sample_ids"],
            "target": prediction["targets"].numpy(),
            "logit": prediction["logits"].numpy(),
            "probability": prediction["logits"].sigmoid().numpy(),
        }
    ).to_csv(stage_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(result["history"]).to_csv(stage_dir / "history.csv", index=False)
    _deduplicate_best_last(
        stage_dir,
        best_epoch=int(result["best_epoch"]),
        last_epoch=int(result["history"][-1]["epoch"]),
    )
    payload = {
        "status": "complete",
        "signature": signature,
        "architecture": architecture,
        "best_epoch": int(result["best_epoch"]),
        "best_checkpoint": best_path.name,
        "checkpoint_sha256": sha256_file(best_path),
        "config_sha256": sha256_file(stage_dir / "config.yaml"),
        "history_sha256": sha256_file(stage_dir / "history.csv"),
        "predictions_sha256": sha256_file(stage_dir / "validation_predictions.csv"),
        "validation": validation,
        "valid_samples": int(prediction["valid_count"]),
        "rejected_samples": int(prediction["rejected_count"]),
        "transfer": transfer,
    }
    write_json(stage_dir / "metrics.json", payload)
    write_json(stage_dir / "status.json", {"status": "complete", "signature": signature})
    return payload


def evaluation_stage(
    context: Mapping[str, Any], stage_dir: Path, settings: Mapping[str, Any]
) -> dict[str, Any]:
    definition = {
        "kind": "incumbent_validation_recompute",
        "model": context["base_model"],
        "data": dict(settings),
        "parent_sha256": context["parent_record"].checkpoint_sha256,
        "inputs": context["input_hashes"],
    }
    signature = phase8_fingerprint(definition)
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("incumbent evaluation signature does not match")
        return payload
    stage_dir.mkdir(parents=True, exist_ok=True)
    module, _ = make_data(context, settings, train_limit=None, val_limit=None)
    model_config = dict(context["base_model"])
    model_config["pretrained"] = False
    model = create_hosp_bonenet(model_config, context["schema"])
    model.load_state_dict(context["parent_state"], strict=True)
    trainer = BinaryTrainer(
        model,
        model_alias="hosp_bonenet",
        resolved_config={
            "epochs": 1,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
            "amp": bool(context["amp"]),
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 1,
            "seed": int(context["seed"]),
            "selection_metric": "AUROC",
        },
        checkpoint_dir=stage_dir / "unused",
        pos_weight=class_weight_from_frame(context["full_fit_frame"]),
        device=context["device"],
    )
    prediction = trainer.predict(module.loader("val"))
    validation = binary_classification_metrics(prediction["logits"], prediction["targets"])
    pd.DataFrame(
        {
            "sample_id": prediction["sample_ids"],
            "target": prediction["targets"].numpy(),
            "logit": prediction["logits"].numpy(),
            "probability": prediction["logits"].sigmoid().numpy(),
        }
    ).to_csv(stage_dir / "validation_predictions.csv", index=False)
    payload = {
        "status": "complete",
        "signature": signature,
        "architecture": "B2",
        "validation": validation,
        "checkpoint_sha256": context["parent_record"].checkpoint_sha256,
    }
    write_json(metrics_path, payload)
    return payload


def base_hparams() -> dict[str, Any]:
    return {
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        "dropout": 0.2,
        "label_smoothing": 0.0,
        "gradient_accumulation_steps": 2,
    }


def base_settings() -> dict[str, Any]:
    return {
        "image_size": 384,
        "batch_size": 4,
        "cache_enabled": True,
        "balance_strategy": "weighted_bce",
        "augmentation_profile": "current",
        "num_workers": 0,
    }


__all__ = [
    "base_hparams",
    "base_settings",
    "evaluation_stage",
    "make_data",
    "seed_everything",
    "training_stage",
]
