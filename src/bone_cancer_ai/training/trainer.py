"""Shared masked binary trainer for all Phase 5 baselines."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.training.amp import autocast_context, build_grad_scaler
from bone_cancer_ai.training.callbacks import EarlyStopping
from bone_cancer_ai.training.checkpointing import load_checkpoint, save_checkpoint
from bone_cancer_ai.training.optimizer import build_optimizer
from bone_cancer_ai.training.scheduler import build_scheduler


class BinaryTrainer:
    """Train one-logit classifiers with one fairness-preserving code path."""

    def __init__(
        self,
        model: nn.Module,
        *,
        model_alias: str,
        resolved_config: Mapping[str, Any],
        checkpoint_dir: Path,
        pos_weight: float | torch.Tensor,
        device: str | torch.device,
        resume_from: Path | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not model_alias:
            raise ValueError("model_alias must be non-empty")
        try:
            self.resolved_config = json.loads(
                json.dumps(dict(resolved_config), allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("resolved_config must be strict JSON-safe data") from error
        self.epochs = self._positive_int("epochs", 40)
        self.accumulation_steps = self._positive_int("gradient_accumulation_steps", 1)
        self.gradient_clip_norm = self._positive_float("gradient_clip_norm", 1.0)
        self.seed = self._integer("seed", 42)
        self.selection_metric = str(self.resolved_config.get("selection_metric", "AUROC"))
        if self.selection_metric != "AUROC":
            raise ValueError("validation AUROC is the only supported selection metric")

        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("trainer device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.model_alias = model_alias
        self.checkpoint_dir = Path(checkpoint_dir)

        weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        if weight.ndim != 0 or not bool(torch.isfinite(weight)) or float(weight) <= 0:
            raise ValueError("pos_weight must be a finite positive scalar")
        self.pos_weight = weight.to(self.device)
        self.amp_requested = bool(self.resolved_config.get("amp", True))
        self.optimizer = build_optimizer(
            self.model,
            learning_rate=self._positive_float("learning_rate", 1e-4),
            weight_decay=self._non_negative_float("weight_decay", 1e-4),
        )
        warmup_epochs = self._integer("warmup_epochs", 2)
        self.scheduler = build_scheduler(
            self.optimizer,
            total_epochs=self.epochs,
            warmup_epochs=warmup_epochs,
        )
        self.scaler = build_grad_scaler(self.device, self.amp_requested)
        self.early_stopping = EarlyStopping(
            patience=self._positive_int("early_stopping_patience", 8)
        )
        self.history: list[dict[str, int | float]] = []
        self.start_epoch = 1
        if resume_from is not None:
            metadata = load_checkpoint(
                Path(resume_from),
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                early_stopping=self.early_stopping,
                expected_model_alias=self.model_alias,
                map_location=self.device,
                expected_resolved_config=self.resolved_config,
                expected_selection_metric=self.selection_metric,
            )
            self.history = [dict(record) for record in metadata["history"]]
            self.start_epoch = int(metadata["epoch"]) + 1

    def _integer(self, key: str, default: int) -> int:
        value = self.resolved_config.get(key, default)
        if type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        return value

    def _positive_int(self, key: str, default: int) -> int:
        value = self._integer(key, default)
        if value < 1:
            raise ValueError(f"{key} must be positive")
        return value

    def _finite_float(self, key: str, default: float) -> float:
        try:
            value = float(self.resolved_config.get(key, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    def _positive_float(self, key: str, default: float) -> float:
        value = self._finite_float(key, default)
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value

    def _non_negative_float(self, key: str, default: float) -> float:
        value = self._finite_float(key, default)
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
        return value

    @staticmethod
    def _seed_dataset(dataset: object, seed: int, seen: set[int] | None = None) -> None:
        if dataset is None:
            return
        seen = set() if seen is None else seen
        if id(dataset) in seen:
            return
        seen.add(id(dataset))
        transform = getattr(dataset, "transform", None)
        set_seed = getattr(transform, "set_seed", None)
        if callable(set_seed):
            set_seed(seed)
        BinaryTrainer._seed_dataset(getattr(dataset, "dataset", None), seed, seen)
        for nested in getattr(dataset, "datasets", ()):
            BinaryTrainer._seed_dataset(nested, seed, seen)

    def _seed_epoch(self, loader: Iterable[Mapping[str, Any]], epoch: int) -> None:
        workers = int(getattr(loader, "num_workers", 0) or 0)
        if workers > 0 and bool(getattr(loader, "persistent_workers", False)):
            raise ValueError(
                "persistent_workers must be false so stochastic workers can be reseeded per epoch"
            )
        seed = (self.seed + epoch - 1) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        batch_sampler = getattr(loader, "batch_sampler", None)
        owners = (
            loader,
            getattr(loader, "sampler", None),
            batch_sampler,
            getattr(batch_sampler, "sampler", None),
        )
        seeded_generators: set[int] = set()
        for owner in owners:
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and id(generator) not in seeded_generators:
                generator.manual_seed(seed)
                seeded_generators.add(id(generator))
        self._seed_dataset(getattr(loader, "dataset", None), seed)

    @staticmethod
    def _batch_size(images: torch.Tensor) -> int:
        if not isinstance(images, torch.Tensor) or images.ndim < 1:
            raise ValueError("batch image must be a tensor with a batch dimension")
        if images.shape[0] < 1:
            raise ValueError("batch image must not be empty")
        return int(images.shape[0])

    @staticmethod
    def _one_dimensional_tensor(value: object, *, name: str, size: int) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != size:
            raise ValueError(f"batch {name} must be a one-dimensional tensor of length {size}")
        return value

    def _select_batch(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
        if not isinstance(batch, Mapping):
            raise ValueError("loader batches must be mappings")
        try:
            images = batch["image"]
            labels = batch["binary_label"]
            masks = batch["valid_task_mask"]
            metadata = batch["metadata"]
            sample_mask = masks["sample"]
            binary_mask = masks["binary"]
            sample_ids = metadata["sample_id"]
        except (KeyError, TypeError) as error:
            raise ValueError("batch is missing the binary training contract") from error
        size = self._batch_size(images)
        labels = self._one_dimensional_tensor(labels, name="binary_label", size=size)
        sample_mask = self._one_dimensional_tensor(
            sample_mask, name="valid_task_mask.sample", size=size
        )
        binary_mask = self._one_dimensional_tensor(
            binary_mask, name="valid_task_mask.binary", size=size
        )
        if isinstance(sample_ids, str) or not isinstance(sample_ids, Sequence):
            raise ValueError("batch metadata.sample_id must be a sequence")
        if len(sample_ids) != size:
            raise ValueError("batch metadata.sample_id length must match the batch")
        valid = sample_mask.to(torch.bool) & binary_mask.to(torch.bool)
        rejected = size - int(valid.sum().item())
        if not bool(valid.any()):
            return images[:0].to(self.device), labels[:0].to(self.device), [], rejected
        selected_labels = labels[valid]
        if not bool(((selected_labels == 0) | (selected_labels == 1)).all()):
            raise ValueError("valid binary labels must contain only 0 and 1")
        indices = valid.nonzero(as_tuple=False).flatten().tolist()
        selected_ids = [str(sample_ids[index]) for index in indices]
        selected_images = images[valid].to(self.device, non_blocking=self.device.type == "cuda")
        selected_targets = selected_labels.to(
            self.device, dtype=torch.float32, non_blocking=self.device.type == "cuda"
        )
        return selected_images, selected_targets, selected_ids, rejected

    @staticmethod
    def _normalize_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise ValueError("model must return a tensor of binary logits")
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.ndim != 1 or logits.shape != targets.shape:
            raise ValueError("model must return one binary logit per valid sample")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("model produced non-finite logits")
        return logits

    def _optimizer_step(self, valid_samples: int) -> float:
        self.scaler.unscale_(self.optimizer)
        if valid_samples < 1:
            raise RuntimeError("optimizer step requires valid accumulated samples")
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(valid_samples)
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("training produced missing or non-finite gradients")
        raw_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        raw_norm_value = float(raw_norm.detach().cpu())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
            raise FloatingPointError("optimizer produced non-finite parameters")
        self.optimizer.zero_grad(set_to_none=True)
        return min(raw_norm_value, self.gradient_clip_norm)

    def _train_epoch(self, loader: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        valid_samples = 0
        rejected_samples = 0
        optimizer_steps = 0
        pending_batches = 0
        pending_valid_samples = 0
        maximum_gradient_norm = 0.0
        for batch in loader:
            images, targets, _sample_ids, rejected = self._select_batch(batch)
            rejected_samples += rejected
            if targets.numel() == 0:
                continue
            with autocast_context(self.device, self.amp_requested):
                logits = self._normalize_logits(self.model(images), targets)
                loss_sum = functional.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=self.pos_weight,
                    reduction="sum",
                )
            if not bool(torch.isfinite(loss_sum)):
                raise FloatingPointError("training produced a non-finite loss")
            count = int(targets.numel())
            total_loss += float(loss_sum.detach().cpu())
            valid_samples += count
            self.scaler.scale(loss_sum).backward()
            pending_batches += 1
            pending_valid_samples += count
            if pending_batches == self.accumulation_steps:
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    self._optimizer_step(pending_valid_samples),
                )
                optimizer_steps += 1
                pending_batches = 0
                pending_valid_samples = 0
        if valid_samples == 0:
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("training epoch has no valid binary samples")
        if pending_batches:
            maximum_gradient_norm = max(
                maximum_gradient_norm,
                self._optimizer_step(pending_valid_samples),
            )
            optimizer_steps += 1
        return {
            "train_loss": float(total_loss / valid_samples),
            "train_valid_samples": valid_samples,
            "train_rejected_samples": rejected_samples,
            "optimizer_steps": optimizer_steps,
            "gradient_norm": float(maximum_gradient_norm),
        }

    def predict(self, loader: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Return ordered masked CPU predictions without computing a selection metric."""

        module_states = [(module, module.training) for module in self.model.modules()]
        logits_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        sample_ids: list[str] = []
        rejected_count = 0
        try:
            self.model.eval()
            with torch.inference_mode():
                for batch in loader:
                    images, targets, selected_ids, rejected = self._select_batch(batch)
                    rejected_count += rejected
                    if targets.numel() == 0:
                        continue
                    with autocast_context(self.device, self.amp_requested):
                        batch_logits = self._normalize_logits(self.model(images), targets)
                    logits_parts.append(batch_logits.detach().to(device="cpu", dtype=torch.float32))
                    target_parts.append(targets.detach().to(device="cpu", dtype=torch.long))
                    sample_ids.extend(selected_ids)
        finally:
            for module, training in module_states:
                module.training = training
        if not logits_parts:
            raise ValueError("prediction has no valid binary samples")
        logits = torch.cat(logits_parts)
        targets = torch.cat(target_parts)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("prediction produced non-finite logits")
        return {
            "logits": logits,
            "targets": targets,
            "sample_ids": sample_ids,
            "valid_count": int(targets.numel()),
            "rejected_count": rejected_count,
        }

    def _validation_epoch(
        self, loader: Iterable[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, int | float]]:
        prediction = self.predict(loader)
        loss = functional.binary_cross_entropy_with_logits(
            prediction["logits"],
            prediction["targets"].to(torch.float32),
            pos_weight=self.pos_weight.cpu(),
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("validation produced a non-finite loss")
        metrics = binary_classification_metrics(prediction["logits"], prediction["targets"])
        auroc = metrics["AUROC"]
        if auroc is None or not math.isfinite(float(auroc)):
            raise ValueError("validation AUROC is undefined")
        return prediction, {
            "val_loss": float(loss),
            "val_AUROC": float(auroc),
            "val_valid_samples": int(prediction["valid_count"]),
            "val_rejected_samples": int(prediction["rejected_count"]),
        }

    def _save(self, filename: str, *, epoch: int, selection_value: float) -> None:
        save_checkpoint(
            self.checkpoint_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            early_stopping=self.early_stopping,
            model_alias=self.model_alias,
            epoch=epoch,
            selection_metric=self.selection_metric,
            selection_value=selection_value,
            resolved_config=self.resolved_config,
            history=self.history,
        )

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Fit using train/validation only and select strictly by validation AUROC."""

        if self.early_stopping.bad_epochs >= self.early_stopping.patience:
            return self._fit_result()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._seed_epoch(train_loader, epoch)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train_record = self._train_epoch(train_loader)
            _prediction, validation_record = self._validation_epoch(val_loader)
            selection_value = float(validation_record["val_AUROC"])
            previous_best = self.early_stopping.best_value
            should_stop = self.early_stopping.step(selection_value, epoch=epoch)
            improved = previous_best is None or selection_value > previous_best
            record: dict[str, int | float] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                **train_record,
                **validation_record,
            }
            self.history.append(record)
            self.scheduler.step()
            self._save("last.pt", epoch=epoch, selection_value=selection_value)
            if improved:
                self._save("best.pt", epoch=epoch, selection_value=selection_value)
            if should_stop:
                break
        return self._fit_result()

    def _fit_result(self) -> dict[str, Any]:
        if self.early_stopping.best_epoch is None or self.early_stopping.best_value is None:
            raise RuntimeError("training completed without a selected validation checkpoint")
        return {
            "history": [dict(record) for record in self.history],
            "best_epoch": self.early_stopping.best_epoch,
            "best_AUROC": self.early_stopping.best_value,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
            "last_checkpoint": str(self.checkpoint_dir / "last.pt"),
        }
