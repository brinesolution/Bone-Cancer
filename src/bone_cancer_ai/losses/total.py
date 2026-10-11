"""Configurable HOSP-BoneNet multi-task loss."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
from torch import nn

from .classification import (
    differentiable_zero,
    masked_binary_cross_entropy,
    masked_cross_entropy,
)
from .concepts import masked_concept_loss
from .grounding import spatial_grounding_loss
from .hierarchy import hierarchy_consistency_loss
from .segmentation import dice_bce_segmentation_loss


@dataclass(frozen=True)
class LossWeights:
    binary: float = 1.0
    subtype: float = 0.0
    risk: float = 0.0
    segmentation: float = 0.0
    concepts: float = 0.0
    grounding: float = 0.0
    hierarchy: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"loss weight {field.name} must be finite and non-negative")


class HOSPLoss(nn.Module):
    """Compose only loss terms backed by explicit predictions and supervision."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        positive_subtype_indices: tuple[int, ...] | None = None,
        subtype_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.positive_subtype_indices = positive_subtype_indices
        if subtype_class_weights is not None:
            if (
                subtype_class_weights.ndim != 1
                or not subtype_class_weights.is_floating_point()
                or not bool(torch.isfinite(subtype_class_weights).all())
                or not bool((subtype_class_weights > 0).all())
            ):
                raise ValueError("subtype_class_weights must be finite and positive")
            self.register_buffer("subtype_class_weights", subtype_class_weights.detach().clone())
        else:
            self.subtype_class_weights = None

    @staticmethod
    def _task_mask(batch: Mapping[str, object], task: str) -> torch.Tensor | None:
        masks = batch.get("valid_task_mask")
        if not isinstance(masks, Mapping):
            return None
        sample = masks.get("sample")
        task_mask = masks.get(task)
        if not isinstance(sample, torch.Tensor) or not isinstance(task_mask, torch.Tensor):
            return None
        if sample.ndim != 1:
            raise ValueError("valid_task_mask.sample must have shape [batch]")
        if task_mask.ndim == 1:
            if task_mask.shape != sample.shape:
                raise ValueError(f"valid_task_mask.{task} must match the sample mask")
            return sample.to(torch.bool) & task_mask.to(device=sample.device, dtype=torch.bool)
        if task == "concepts" and task_mask.ndim == 2 and task_mask.shape[0] == sample.shape[0]:
            return sample.to(torch.bool).unsqueeze(1) & task_mask.to(
                device=sample.device, dtype=torch.bool
            )
        raise ValueError(f"valid_task_mask.{task} has an invalid shape")

    @staticmethod
    def _has_valid(mask: torch.Tensor | None) -> bool:
        return mask is not None and bool(mask.any())

    @staticmethod
    def _require_tensor(
        value: object,
        *,
        prediction_name: str,
        task_name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"{prediction_name} is required when valid {task_name} labels are present"
            )
        return value

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        batch: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(predictions, Mapping) or not isinstance(batch, Mapping):
            raise TypeError("predictions and batch must be mappings")
        prediction_tensors = tuple(
            value for value in predictions.values() if isinstance(value, torch.Tensor)
        )
        if not prediction_tensors:
            raise ValueError("predictions must contain at least one tensor")
        zero = differentiable_zero(*prediction_tensors)
        losses = {
            "binary": zero,
            "subtype": zero,
            "risk": zero,
            "segmentation": zero,
            "segmentation_bce": zero,
            "segmentation_dice": zero,
            "concepts": zero,
            "grounding": zero,
            "hierarchy": zero,
        }
        binary_logits = predictions.get("binary_logits")
        binary_targets = batch.get("binary_label")
        binary_mask = self._task_mask(batch, "binary")
        if self.weights.binary > 0 and self._has_valid(binary_mask):
            binary_logits = self._require_tensor(
                binary_logits,
                prediction_name="binary_logits",
                task_name="binary",
            )
            binary_targets = self._require_tensor(
                binary_targets,
                prediction_name="binary_label",
                task_name="binary",
            )
        if (
            self.weights.binary > 0
            and isinstance(binary_logits, torch.Tensor)
            and isinstance(binary_targets, torch.Tensor)
            and binary_mask is not None
        ):
            losses["binary"] = masked_binary_cross_entropy(
                binary_logits, binary_targets, binary_mask
            )
        for task, prediction_key, target_key in (
            ("subtype", "subtype_logits", "subtype_label"),
            ("risk", "risk_logits", "risk_label"),
        ):
            logits = predictions.get(prediction_key)
            targets = batch.get(target_key)
            valid_mask = self._task_mask(batch, task)
            if getattr(self.weights, task) > 0 and self._has_valid(valid_mask):
                logits = self._require_tensor(
                    logits,
                    prediction_name=prediction_key,
                    task_name=task,
                )
                targets = self._require_tensor(
                    targets,
                    prediction_name=target_key,
                    task_name=task,
                )
            if (
                getattr(self.weights, task) > 0
                and isinstance(logits, torch.Tensor)
                and isinstance(targets, torch.Tensor)
                and valid_mask is not None
            ):
                losses[task] = masked_cross_entropy(
                    logits,
                    targets,
                    valid_mask,
                    class_weights=(
                        self.subtype_class_weights if task == "subtype" else None
                    ),
                )
        lesion_logits = predictions.get("lesion_logits")
        mask_targets = batch.get("mask")
        mask_valid = self._task_mask(batch, "mask")
        if self.weights.segmentation > 0 and self._has_valid(mask_valid):
            lesion_logits = self._require_tensor(
                lesion_logits,
                prediction_name="lesion_logits",
                task_name="mask",
            )
            mask_targets = self._require_tensor(
                mask_targets,
                prediction_name="mask",
                task_name="mask",
            )
        if (
            self.weights.segmentation > 0
            and isinstance(lesion_logits, torch.Tensor)
            and isinstance(mask_targets, torch.Tensor)
            and mask_valid is not None
        ):
            segmentation = dice_bce_segmentation_loss(
                lesion_logits, mask_targets, mask_valid
            )
            losses["segmentation"] = segmentation.total
            losses["segmentation_bce"] = segmentation.bce
            losses["segmentation_dice"] = segmentation.dice
        concept_logits = predictions.get("concept_logits")
        concept_targets = batch.get("concept_labels")
        concept_valid = self._task_mask(batch, "concepts")
        if self.weights.concepts > 0 and self._has_valid(concept_valid):
            concept_logits = self._require_tensor(
                concept_logits,
                prediction_name="concept_logits",
                task_name="concept",
            )
            concept_targets = self._require_tensor(
                concept_targets,
                prediction_name="concept_labels",
                task_name="concept",
            )
        if (
            self.weights.concepts > 0
            and isinstance(concept_logits, torch.Tensor)
            and isinstance(concept_targets, torch.Tensor)
            and concept_valid is not None
        ):
            losses["concepts"] = masked_concept_loss(
                concept_logits, concept_targets, concept_valid
            )
        concept_maps = predictions.get("concept_maps")
        grounding_is_valid = self._has_valid(concept_valid) and self._has_valid(mask_valid)
        if self.weights.grounding > 0 and grounding_is_valid:
            concept_maps = self._require_tensor(
                concept_maps,
                prediction_name="concept_maps",
                task_name="concept and mask",
            )
            concept_targets = self._require_tensor(
                concept_targets,
                prediction_name="concept_labels",
                task_name="concept and mask",
            )
            mask_targets = self._require_tensor(
                mask_targets,
                prediction_name="mask",
                task_name="concept and mask",
            )
        if (
            self.weights.grounding > 0
            and isinstance(concept_maps, torch.Tensor)
            and isinstance(mask_targets, torch.Tensor)
            and isinstance(concept_targets, torch.Tensor)
            and concept_valid is not None
            and mask_valid is not None
        ):
            losses["grounding"] = spatial_grounding_loss(
                concept_maps,
                mask_targets,
                concept_targets,
                concept_valid,
                mask_valid,
            )
        subtype_logits = predictions.get("subtype_logits")
        subtype_valid = self._task_mask(batch, "subtype")
        if self.weights.hierarchy > 0 and self._has_valid(subtype_valid):
            binary_logits = self._require_tensor(
                binary_logits,
                prediction_name="binary_logits",
                task_name="subtype hierarchy",
            )
            subtype_logits = self._require_tensor(
                subtype_logits,
                prediction_name="subtype_logits",
                task_name="subtype hierarchy",
            )
            if self.positive_subtype_indices is None:
                raise ValueError(
                    "positive_subtype_indices are required for valid hierarchy labels"
                )
        if (
            self.weights.hierarchy > 0
            and isinstance(binary_logits, torch.Tensor)
            and isinstance(subtype_logits, torch.Tensor)
            and subtype_valid is not None
            and self.positive_subtype_indices is not None
        ):
            losses["hierarchy"] = hierarchy_consistency_loss(
                binary_logits,
                subtype_logits,
                subtype_valid,
                self.positive_subtype_indices,
            )
        losses["total"] = (
            self.weights.binary * losses["binary"]
            + self.weights.subtype * losses["subtype"]
            + self.weights.risk * losses["risk"]
            + self.weights.segmentation * losses["segmentation"]
            + self.weights.concepts * losses["concepts"]
            + self.weights.grounding * losses["grounding"]
            + self.weights.hierarchy * losses["hierarchy"]
        )
        return losses
