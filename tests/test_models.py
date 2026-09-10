from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from bone_cancer_ai.models.baselines.factory import (
    count_trainable_parameters,
    create_baseline,
)
from bone_cancer_ai.models.registry import BASELINE_REGISTRY
from bone_cancer_ai.utils.config import load_config

EXPECTED_MODELS = {
    "densenet121": "densenet121",
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def test_registry_exposes_only_approved_frozen_baselines() -> None:
    assert {name: spec.timm_name for name, spec in BASELINE_REGISTRY.items()} == EXPECTED_MODELS

    with pytest.raises(TypeError):
        BASELINE_REGISTRY["other"] = BASELINE_REGISTRY["densenet121"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        BASELINE_REGISTRY["densenet121"].timm_name = "other"  # type: ignore[misc]


def test_factory_rejects_unknown_baseline_name() -> None:
    with pytest.raises(ValueError, match="Unknown baseline 'resnet50'"):
        create_baseline("resnet50", pretrained=False, image_size=384)


@pytest.mark.parametrize("name", EXPECTED_MODELS)
def test_each_baseline_returns_finite_binary_logits(name: str) -> None:
    model = create_baseline(name, pretrained=False, image_size=384)

    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 384, 384))

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert count_trainable_parameters(model) > 0


def test_baseline_configs_share_the_complete_common_training_config() -> None:
    config_dir = Path(__file__).parents[1] / "configs"

    for alias in EXPECTED_MODELS:
        assert load_config(config_dir / f"baseline_{alias}.yaml") == {
            "model": alias,
            "common_config": "train.yaml",
        }

    assert load_config(config_dir / "train.yaml") == {
        "data_config": "data.yaml",
        "output_root": "experiments/baselines",
        "image_size": 384,
        "batch_size": 8,
        "epochs": 40,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "amp": True,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 8,
        "seed": 42,
        "device": "cuda",
        "allow_cpu_training": False,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "selection_split": "val",
        "selection_metric": "AUROC",
    }
