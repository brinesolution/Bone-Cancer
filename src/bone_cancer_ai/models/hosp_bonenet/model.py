"""HOSP-BoneNet architecture and configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

import torch
from torch import nn
from torch.nn import functional as F

from .concept_head import ConceptHead
from .encoder import TimmFeatureEncoder
from .fusion import GatedFusion
from .hierarchy import HierarchicalHeads
from .localization_head import LocalizationHead
from .output import HOSPOutput
from .pooling import GlobalPool, MaskWeightedPool


class FeatureEncoder(Protocol):
    """Structural interface accepted for encoder injection in tests and experiments."""

    feature_channels: tuple[int, ...]
    feature_reductions: tuple[int, ...]

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


@dataclass(frozen=True)
class HOSPConfig:
    """Validated architecture switches for HOSP-BoneNet."""

    backbone: str = "convnext_tiny"
    pretrained: bool = True
    feature_dim: int = 256
    pyramid_channels: int = 128
    dropout: float = 0.2
    pooling: str = "adaptive_avg"
    localization_enabled: bool = True
    spatial_path: str = "auto"
    subtype_enabled: bool = False
    num_subtypes: int = 0
    risk_enabled: bool = False
    num_risk_classes: int = 0
    concepts_enabled: bool = False
    num_concepts: int = 0

    def __post_init__(self) -> None:
        if not self.backbone:
            raise ValueError("backbone must be nonempty")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.pyramid_channels <= 0:
            raise ValueError("pyramid_channels must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pooling not in {"adaptive_avg", "avg", "average", "gem"}:
            raise ValueError("pooling must be 'adaptive_avg' or 'gem'")
        if self.spatial_path not in {"auto", "none", "attention", "gated"}:
            raise ValueError("spatial_path must be auto, none, attention, or gated")
        if not self.localization_enabled and self.spatial_path in {"attention", "gated"}:
            raise ValueError("spatial_path requires the localization branch")
        if self.localization_enabled and self.spatial_path == "none":
            raise ValueError("spatial_path none cannot enable the localization branch")
        if self.subtype_enabled and self.num_subtypes < 2:
            raise ValueError("num_subtypes must be at least 2 when subtype is enabled")
        if self.risk_enabled and self.num_risk_classes < 2:
            raise ValueError("num_risk_classes must be at least 2 when risk is enabled")
        if self.concepts_enabled and self.num_concepts < 1:
            raise ValueError("num_concepts must be positive when concepts are enabled")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> HOSPConfig:
        """Build a strict config while accepting the YAML model identity field."""

        model_name = values.get("name", "hosp_bonenet")
        if model_name != "hosp_bonenet":
            raise ValueError("model.name must be 'hosp_bonenet'")
        allowed = {field.name for field in fields(cls)}
        unexpected = set(values) - allowed - {"name"}
        if unexpected:
            raise ValueError(f"unknown HOSP-BoneNet model fields: {sorted(unexpected)}")
        return cls(**{key: value for key, value in values.items() if key in allowed})


class HOSPBoneNet(nn.Module):
    """Hierarchical spatial model with configuration-gated auxiliary outputs."""

    def __init__(self, config: HOSPConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.spatial_path = (
            "gated" if config.localization_enabled else "none"
        ) if config.spatial_path == "auto" else config.spatial_path
        self.encoder = encoder or TimmFeatureEncoder(config.backbone, config.pretrained)
        channels = tuple(int(value) for value in self.encoder.feature_channels)
        reductions = tuple(int(value) for value in self.encoder.feature_reductions)
        if not channels or len(channels) != len(reductions):
            raise ValueError("encoder must expose matching feature_channels and feature_reductions")
        if any(value <= 0 for value in channels) or any(value <= 0 for value in reductions):
            raise ValueError("encoder feature metadata must contain positive values")

        self.feature_projection = nn.Conv2d(channels[-1], config.feature_dim, kernel_size=1)
        self.global_pool = GlobalPool(config.pooling)
        self.localization_head = (
            LocalizationHead(channels, config.pyramid_channels)
            if config.localization_enabled
            else None
        )
        self.local_pool = MaskWeightedPool() if self.spatial_path == "gated" else None
        self.fusion = GatedFusion(config.feature_dim) if self.spatial_path == "gated" else None
        self.concept_head = (
            ConceptHead(config.feature_dim, config.num_concepts)
            if config.concepts_enabled
            else None
        )
        head_input_dim = config.feature_dim + (
            config.num_concepts if config.concepts_enabled else 0
        )
        self.heads = HierarchicalHeads(
            head_input_dim,
            dropout=config.dropout,
            num_subtypes=config.num_subtypes if config.subtype_enabled else None,
            num_risk_classes=(
                config.num_risk_classes if config.risk_enabled else None
            ),
        )

    def forward(self, images: torch.Tensor) -> HOSPOutput:
        """Predict configured outputs from images without requiring labels."""

        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        if images.shape[1] != 3:
            raise ValueError("images must contain three channels")

        features = tuple(self.encoder(images))
        expected_stages = len(self.encoder.feature_channels)
        if len(features) != expected_stages:
            raise ValueError(
                f"encoder returned {len(features)} feature maps; expected {expected_stages}"
            )
        deepest = self.feature_projection(features[-1])
        global_features = self.global_pool(deepest)
        output: HOSPOutput = {"global_features": global_features}

        base_features = global_features
        if self.localization_head is not None:
            lesion_logits = self.localization_head(features, images.shape[-2:])
            output["lesion_logits"] = lesion_logits
            if self.spatial_path == "attention":
                attention = F.interpolate(
                    lesion_logits.sigmoid(),
                    size=deepest.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                base_features = self.global_pool(deepest * (1.0 + attention))
            elif self.spatial_path == "gated":
                assert self.local_pool is not None and self.fusion is not None
                local_features = self.local_pool(deepest, lesion_logits)
                base_features = self.fusion(global_features, local_features)
                output["local_features"] = local_features

        fused_features = base_features
        if self.concept_head is not None:
            concept_logits, concept_maps = self.concept_head(deepest, images.shape[-2:])
            fused_features = torch.cat((base_features, concept_logits.sigmoid()), dim=1)
            output["concept_logits"] = concept_logits
            output["concept_maps"] = concept_maps

        output["fused_features"] = fused_features
        output.update(self.heads(fused_features))
        return output


def _configured_count(section: Mapping[str, Any], field: str) -> int:
    values = section.get(field, ())
    if isinstance(values, Mapping | Sequence) and not isinstance(values, str | bytes):
        return len(values)
    raise ValueError(f"label schema {field} must be a class/name collection")


def create_hosp_bonenet(
    model_config: Mapping[str, Any],
    label_schema: Mapping[str, Any],
    *,
    encoder: nn.Module | None = None,
) -> HOSPBoneNet:
    """Create a model whose optional dimensions come from enabled label-schema gates."""

    binary_schema = label_schema.get("binary", {})
    if not isinstance(binary_schema, Mapping) or not bool(binary_schema.get("enabled")):
        raise ValueError("binary label schema must be enabled")
    resolved = dict(model_config)
    task_fields = (
        ("subtype", "subtype_enabled", "classes", "num_subtypes"),
        ("risk", "risk_enabled", "classes", "num_risk_classes"),
        ("concepts", "concepts_enabled", "names", "num_concepts"),
    )
    for task, enabled_field, values_field, count_field in task_fields:
        enabled = bool(resolved.get(enabled_field, False))
        section = label_schema.get(task, {})
        if enabled:
            if not isinstance(section, Mapping) or not bool(section.get("enabled")):
                raise ValueError(f"{task} head requested while its label schema is disabled")
            resolved[count_field] = _configured_count(section, values_field)
        else:
            resolved[count_field] = 0
    return HOSPBoneNet(HOSPConfig.from_mapping(resolved), encoder=encoder)


__all__ = ["HOSPBoneNet", "HOSPConfig", "create_hosp_bonenet"]
