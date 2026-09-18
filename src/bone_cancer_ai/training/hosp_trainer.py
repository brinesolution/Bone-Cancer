"""Masked multi-task trainer used by the BTXRD subtype migration run."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.evaluation.hosp_btxrd import subtype_classification_metrics
from bone_cancer_ai.losses.total import HOSPLoss, LossWeights
from bone_cancer_ai.training.amp import autocast_context, build_grad_scaler
from bone_cancer_ai.training.callbacks import EarlyStopping
from bone_cancer_ai.training.checkpointing import load_checkpoint, save_checkpoint
from bone_cancer_ai.training.optimizer import build_optimizer
from bone_cancer_ai.training.scheduler import build_scheduler


class HOSPMultiTaskTrainer:
    """Train binary and subtype heads with explicit per-task validity masks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        model_alias: str,
        resolved_config: Mapping[str, Any],
        checkpoint_dir: Path,
        device: str | torch.device,
        resume_from: Path | None = None,
        epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not model_alias:
            raise ValueError("model_alias must be non-empty")
        try:
            self.resolved_config = json.loads(json.dumps(dict(resolved_config), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("resolved_config must be strict JSON-safe data") from error
        self.model_alias = model_alias
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("trainer device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.epochs = self._positive_int("epochs", 30)
        self.accumulation_steps = self._positive_int("gradient_accumulation_steps", 1)
        self.gradient_clip_norm = self._positive_float("gradient_clip_norm", 1.0)
        self.seed = self._integer("seed", 42)
        self.selection_metric = str(self.resolved_config.get("selection_metric", "AUROC"))
        if self.selection_metric != "AUROC":
            raise ValueError("validation AUROC is the only supported selection metric")
        self.amp_requested = bool(self.resolved_config.get("amp", True))
        self.criterion = HOSPLoss(weights=self._loss_weights())
        self.optimizer = build_optimizer(
            self.model,
            learning_rate=self._positive_float("learning_rate", 1e-4),
            weight_decay=self._non_negative_float("weight_decay", 1e-4),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_epochs=self.epochs,
            warmup_epochs=self._integer("warmup_epochs", 2),
        )
        self.scaler = build_grad_scaler(self.device, self.amp_requested)
        self.early_stopping = EarlyStopping(
            patience=self._positive_int("early_stopping_patience", 8),
            min_delta=float(self.resolved_config.get("early_stopping_min_delta", 0.0)),
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_checkpoint_path = self.checkpoint_dir / "best.pt"
        self.last_checkpoint_path = self.checkpoint_dir / "last.pt"
        self.history: list[dict[str, Any]] = []
        self.start_epoch = 1
        self.epoch_callback = epoch_callback
        if resume_from is not None:
            resume_path = Path(resume_from)
            if not (resume_path.parent / "last.pt").is_file():
                raise FileNotFoundError("resume checkpoint directory must contain last.pt")
            metadata = load_checkpoint(
                resume_path,
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
            self.best_checkpoint_path = resume_path.parent / "best.pt"
            self.last_checkpoint_path = resume_path.parent / "last.pt"
            if not self.best_checkpoint_path.is_file():
                if metadata["early_stopping_state"]["best_epoch"] != metadata["epoch"]:
                    raise ValueError("historical best checkpoint is missing")
                self.best_checkpoint_path.write_bytes(resume_path.read_bytes())

    def _loss_weights(self) -> LossWeights:
        raw = self.resolved_config.get("loss_weights", {})
        values = dict(raw) if isinstance(raw, Mapping) else {}
        if "subtype_loss_weight" in self.resolved_config:
            values["subtype"] = self.resolved_config["subtype_loss_weight"]
        return LossWeights(
            binary=float(values.get("binary", 1.0)),
            subtype=float(values.get("subtype", 1.0)),
            risk=float(values.get("risk", 0.0)),
            segmentation=float(values.get("segmentation", 0.0)),
            concepts=float(values.get("concepts", 0.0)),
            grounding=float(values.get("grounding", 0.0)),
            hierarchy=float(values.get("hierarchy", 0.0)),
        )

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

    def _float(self, key: str, default: float) -> float:
        try:
            value = float(self.resolved_config.get(key, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    def _positive_float(self, key: str, default: float) -> float:
        value = self._float(key, default)
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value

    def _non_negative_float(self, key: str, default: float) -> float:
        value = self._float(key, default)
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
        return value

    @staticmethod
    def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=device.type == "cuda")
            if isinstance(value, Mapping):
                return {key: move(item) for key, item in value.items()}
            return value

        return {key: move(value) for key, value in batch.items()}

    def _seed_epoch(self, epoch: int) -> None:
        seed = (self.seed + epoch - 1) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _outputs(model_output: object) -> Mapping[str, torch.Tensor]:
        if not isinstance(model_output, Mapping):
            raise ValueError("HOSP model must return a tensor mapping")
        if not isinstance(model_output.get("binary_logits"), torch.Tensor):
            raise ValueError("HOSP model output must contain binary_logits")
        return model_output

    def _optimizer_step(self, valid_samples: int) -> float:
        self.scaler.unscale_(self.optimizer)
        if valid_samples < 1:
            raise RuntimeError("optimizer step requires valid samples")
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(valid_samples)
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("training produced missing or non-finite gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm, error_if_nonfinite=True
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return float(min(float(norm.detach().cpu()), self.gradient_clip_norm))

    def _epoch(self, loader: Iterable[Mapping[str, Any]], *, train: bool) -> dict[str, Any]:
        self.model.train(train)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        valid_binary = 0
        valid_subtype = 0
        rejected = 0
        pending_valid = 0
        pending_batches = 0
        max_norm = 0.0
        binary_logits: list[torch.Tensor] = []
        binary_targets: list[torch.Tensor] = []
        subtype_logits: list[torch.Tensor] = []
        subtype_targets: list[torch.Tensor] = []
        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            for raw_batch in loader:
                if not isinstance(raw_batch, Mapping):
                    raise ValueError("loader batches must be mappings")
                batch = self._move_batch(raw_batch, self.device)
                masks = batch.get("valid_task_mask")
                if not isinstance(masks, Mapping):
                    raise ValueError("batch is missing valid_task_mask")
                sample_mask = masks["sample"].bool()
                binary_mask = sample_mask & masks["binary"].bool()
                subtype_mask = sample_mask & masks["subtype"].bool()
                rejected += int((~sample_mask).sum().item())
                valid_binary += int(binary_mask.sum().item())
                valid_subtype += int(subtype_mask.sum().item())
                with autocast_context(self.device, self.amp_requested):
                    outputs = self._outputs(self.model(batch["image"]))
                    losses = self.criterion(outputs, batch)
                loss = losses["total"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("training produced a non-finite loss")
                total_loss += float(loss.detach().cpu())
                batches += 1
                if not train:
                    binary_logits.append(
                        outputs["binary_logits"].detach().cpu().reshape(-1)[binary_mask.cpu()]
                    )
                    binary_targets.append(
                        batch["binary_label"].detach().cpu().reshape(-1)[binary_mask.cpu()]
                    )
                    subtype_output = outputs.get("subtype_logits")
                    if isinstance(subtype_output, torch.Tensor) and bool(subtype_mask.any()):
                        cpu_subtype_mask = subtype_mask.cpu()
                        subtype_logits.append(subtype_output.detach().cpu()[cpu_subtype_mask])
                        subtype_targets.append(
                            batch["subtype_label"].detach().cpu()[cpu_subtype_mask]
                        )
                if train:
                    self.scaler.scale(loss).backward()
                    pending_batches += 1
                    # HOSPLoss returns a per-batch mean; average accumulated batch gradients.
                    pending_valid += 1
                    if pending_batches == self.accumulation_steps:
                        max_norm = max(max_norm, self._optimizer_step(pending_valid))
                        pending_batches = 0
                        pending_valid = 0
        if train and pending_batches:
            max_norm = max(max_norm, self._optimizer_step(pending_valid))
        if valid_binary == 0:
            raise ValueError("epoch has no valid binary samples")
        result: dict[str, Any] = {
            "loss": float(total_loss / max(1, batches)),
            "valid_binary": valid_binary,
            "valid_subtype": valid_subtype,
            "rejected": rejected,
            "optimizer_steps": int(
                0 if not train else max(1, math.ceil(batches / self.accumulation_steps))
            ),
            "gradient_norm": max_norm,
        }
        if not train:
            result.update(
                {
                    "binary_logits": torch.cat(binary_logits) if binary_logits else torch.empty(0),
                    "binary_targets": torch.cat(binary_targets)
                    if binary_targets
                    else torch.empty(0, dtype=torch.long),
                    "subtype_logits": torch.cat(subtype_logits)
                    if subtype_logits
                    else torch.empty((0, 0)),
                    "subtype_targets": torch.cat(subtype_targets)
                    if subtype_targets
                    else torch.empty(0, dtype=torch.long),
                }
            )
        return result

    def _save(self, name: str, epoch: int, value: float) -> None:
        save_checkpoint(
            self.checkpoint_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            early_stopping=self.early_stopping,
            model_alias=self.model_alias,
            epoch=epoch,
            selection_metric=self.selection_metric,
            selection_value=value,
            resolved_config=self.resolved_config,
            history=self.history,
        )

    def fit(
        self, train_loader: Iterable[Mapping[str, Any]], val_loader: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Fit and checkpoint on validation AUROC; calibration/test stay outside fit."""

        if self.early_stopping.bad_epochs >= self.early_stopping.patience:
            return self._result()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._seed_epoch(epoch)
            train = self._epoch(train_loader, train=True)
            validation = self._epoch(val_loader, train=False)
            binary_metrics = binary_classification_metrics(
                validation["binary_logits"], validation["binary_targets"]
            )
            auroc = binary_metrics["AUROC"]
            if auroc is None or not math.isfinite(float(auroc)):
                raise ValueError("validation AUROC is undefined")
            record: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "train_loss": train["loss"],
                "train_valid_samples": train["valid_binary"],
                "train_subtype_valid_samples": train["valid_subtype"],
                "train_rejected_samples": train["rejected"],
                "val_loss": validation["loss"],
                "val_AUROC": float(auroc),
                "val_valid_samples": validation["valid_binary"],
                "val_subtype_valid_samples": validation["valid_subtype"],
                "val_rejected_samples": validation["rejected"],
                "gradient_norm": train["gradient_norm"],
            }
            if validation["subtype_logits"].shape[0]:
                record["val_subtype_macro_f1"] = subtype_classification_metrics(
                    validation["subtype_logits"], validation["subtype_targets"]
                )["macro_f1"]
            self.history.append(record)
            previous = self.early_stopping.best_value
            should_stop = self.early_stopping.step(float(auroc), epoch=epoch)
            improved = previous is None or float(auroc) > previous
            self.scheduler.step()
            self._save("last.pt", epoch, float(auroc))
            if improved:
                self._save("best.pt", epoch, float(auroc))
            if self.epoch_callback is not None:
                self.epoch_callback(dict(record))
            if should_stop:
                break
        return self._result()

    def _result(self) -> dict[str, Any]:
        if self.early_stopping.best_epoch is None or self.early_stopping.best_value is None:
            raise RuntimeError("training completed without a validation checkpoint")
        return {
            "history": [dict(item) for item in self.history],
            "best_epoch": self.early_stopping.best_epoch,
            "best_AUROC": self.early_stopping.best_value,
            "best_checkpoint": str(self.best_checkpoint_path),
            "last_checkpoint": str(self.last_checkpoint_path),
        }
