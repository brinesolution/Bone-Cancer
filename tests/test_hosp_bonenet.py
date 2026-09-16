from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from bone_cancer_ai.losses import HOSPLoss, LossWeights
from bone_cancer_ai.models.hosp_bonenet import (
    HOSPBoneNet,
    HOSPConfig,
    create_hosp_bonenet,
)
from bone_cancer_ai.utils.config import load_config


class TinyFeatureEncoder(nn.Module):
    feature_channels = (8, 16, 32)
    feature_reductions = (2, 4, 8)

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU()),
            ]
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = []
        value = images
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        return tuple(features)


def _config(**overrides: object) -> HOSPConfig:
    values: dict[str, object] = {
        "backbone": "convnext_tiny",
        "pretrained": False,
        "feature_dim": 24,
        "pyramid_channels": 16,
        "dropout": 0.0,
        "localization_enabled": True,
    }
    values.update(overrides)
    return HOSPConfig(**values)


def test_binary_only_configuration_omits_every_disabled_output() -> None:
    model = HOSPBoneNet(
        _config(localization_enabled=False, spatial_path="none"),
        encoder=TinyFeatureEncoder(),
    ).eval()

    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 80))

    assert set(output) == {"binary_logits", "global_features", "fused_features"}
    assert output["binary_logits"].shape == (2,)
    assert output["global_features"].shape == (2, 24)
    assert output["fused_features"].shape == (2, 24)
    assert all(torch.isfinite(value).all() for value in output.values())


@pytest.mark.parametrize("spatial_path", ["attention", "gated"])
def test_supported_spatial_paths_route_binary_features(spatial_path: str) -> None:
    model = HOSPBoneNet(
        _config(spatial_path=spatial_path), encoder=TinyFeatureEncoder()
    ).eval()

    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))

    if spatial_path == "attention":
        assert model.fusion is None
        assert model.local_pool is None
        assert "local_features" not in output
        assert not torch.equal(output["fused_features"], output["global_features"])
    else:
        assert model.fusion is not None
        assert model.local_pool is not None
        assert not torch.equal(output["fused_features"], output["global_features"])
        assert not torch.equal(output["fused_features"], output["local_features"])


def test_attention_spatial_path_requires_localization_branch() -> None:
    with pytest.raises(ValueError, match="spatial_path.*localization"):
        _config(localization_enabled=False, spatial_path="attention")


def test_unknown_spatial_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="spatial_path"):
        _config(spatial_path="concatenate")


def test_none_spatial_path_rejects_unused_localization_branch() -> None:
    with pytest.raises(ValueError, match="spatial_path.*localization"):
        _config(localization_enabled=True, spatial_path="none")


def test_attention_path_gives_localization_head_binary_gradient() -> None:
    model = HOSPBoneNet(
        _config(spatial_path="attention"), encoder=TinyFeatureEncoder()
    )

    model(torch.randn(2, 3, 64, 64))["binary_logits"].sum().backward()

    assert model.localization_head is not None
    gradients = [parameter.grad for parameter in model.localization_head.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None)


def test_full_configuration_returns_aligned_configured_outputs() -> None:
    model = HOSPBoneNet(
        _config(
            subtype_enabled=True,
            num_subtypes=4,
            risk_enabled=True,
            num_risk_classes=2,
            concepts_enabled=True,
            num_concepts=3,
        ),
        encoder=TinyFeatureEncoder(),
    ).eval()

    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 80))

    assert set(output) == {
        "binary_logits",
        "subtype_logits",
        "risk_logits",
        "concept_logits",
        "concept_maps",
        "lesion_logits",
        "global_features",
        "local_features",
        "fused_features",
    }
    assert output["subtype_logits"].shape == (2, 4)
    assert output["risk_logits"].shape == (2, 2)
    assert output["concept_logits"].shape == (2, 3)
    assert output["concept_maps"].shape == (2, 3, 64, 80)
    assert output["lesion_logits"].shape == (2, 1, 64, 80)
    assert output["local_features"].shape == (2, 24)
    assert output["fused_features"].shape == (2, 27)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_subtypes", 0, "num_subtypes"),
        ("num_risk_classes", 1, "num_risk_classes"),
        ("num_concepts", 0, "num_concepts"),
    ],
)
def test_enabled_optional_heads_require_valid_output_dimensions(
    field: str, value: int, message: str
) -> None:
    kwargs = {
        "subtype_enabled": field == "num_subtypes",
        "risk_enabled": field == "num_risk_classes",
        "concepts_enabled": field == "num_concepts",
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_binary_gradient_reaches_encoder() -> None:
    encoder = TinyFeatureEncoder()
    model = HOSPBoneNet(_config(), encoder=encoder)
    output = model(torch.randn(2, 3, 64, 64))

    output["binary_logits"].sum().backward()

    gradient = encoder.stages[0][0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_forward_is_an_inference_path_that_requires_no_labels() -> None:
    model = HOSPBoneNet(_config(), encoder=TinyFeatureEncoder()).eval()
    with torch.inference_mode():
        output = model(images=torch.zeros(1, 3, 48, 56))
    assert output["binary_logits"].shape == (1,)
    assert output["lesion_logits"].shape[-2:] == (48, 56)


def test_production_config_enables_only_verified_supervision() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "hosp_bonenet.yaml")
    model = config["model"]
    weights = config["loss_weights"]

    assert config["verified_supervision"] == ["binary"]
    assert model["backbone"] == "convnext_tiny"
    assert model["pretrained"] is True
    assert model["localization_enabled"] is True
    assert model["subtype_enabled"] is False
    assert model["risk_enabled"] is False
    assert model["concepts_enabled"] is False
    assert weights["binary"] > 0
    assert weights["subtype"] == 0
    assert weights["segmentation"] == 0
    assert weights["concepts"] == 0
    assert weights["grounding"] == 0
    assert weights["hierarchy"] == 0


def test_factory_rejects_optional_head_without_schema_gate() -> None:
    model_config = {
        "name": "hosp_bonenet",
        "pretrained": False,
        "subtype_enabled": True,
    }
    schema = {"binary": {"enabled": True}, "subtype": {"enabled": False}}

    with pytest.raises(ValueError, match="subtype.*disabled"):
        create_hosp_bonenet(model_config, schema, encoder=TinyFeatureEncoder())


def test_factory_derives_enabled_output_dimensions_from_schema() -> None:
    model_config = {
        "name": "hosp_bonenet",
        "pretrained": False,
        "feature_dim": 24,
        "pyramid_channels": 16,
        "subtype_enabled": True,
        "risk_enabled": True,
        "concepts_enabled": True,
    }
    schema = {
        "binary": {"enabled": True},
        "subtype": {"enabled": True, "classes": ["a", "b", "c"]},
        "risk": {"enabled": True, "classes": ["low", "high"]},
        "concepts": {"enabled": True, "names": ["x", "y"]},
    }

    model = create_hosp_bonenet(model_config, schema, encoder=TinyFeatureEncoder()).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 32, 32))

    assert output["subtype_logits"].shape == (1, 3)
    assert output["risk_logits"].shape == (1, 2)
    assert output["concept_logits"].shape == (1, 2)


def test_cpu_autocast_forward_is_finite() -> None:
    model = HOSPBoneNet(_config(), encoder=TinyFeatureEncoder()).eval()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(torch.randn(2, 3, 64, 64))
    assert all(torch.isfinite(value.float()).all() for value in output.values())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_default_convnext_model_works_in_cuda_amp() -> None:
    model = HOSPBoneNet(_config()).cuda().train()
    images = torch.randn(1, 3, 384, 384, device="cuda")
    batch = {
        "binary_label": torch.tensor([1], device="cuda"),
        "valid_task_mask": {
            "sample": torch.tensor([True], device="cuda"),
            "binary": torch.tensor([True], device="cuda"),
        },
    }

    with torch.autocast("cuda", dtype=torch.float16):
        output = model(images)
        loss = HOSPLoss(LossWeights(binary=1.0))(output, batch)["total"]
    loss.backward()

    assert output["lesion_logits"].shape == (1, 1, 384, 384)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
