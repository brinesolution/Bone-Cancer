from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from bone_cancer_ai.evaluation.hosp_btxrd import (
    MulticlassConformalCalibrator,
    subtype_classification_metrics,
)
from bone_cancer_ai.training.hosp_trainer import HOSPMultiTaskTrainer


class TinyMultiTask(nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.subtype = nn.Linear(3, classes)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        value = images.mean(dim=(2, 3))
        return {
            "binary_logits": self.linear(value).squeeze(1),
            "subtype_logits": self.subtype(value),
        }


def _batch(seed: int = 0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(4, 3, 8, 8, generator=generator)
    return {
        "image": images,
        "binary_label": torch.tensor([0, 1, 1, 0]),
        "subtype_label": torch.tensor([-1, 0, 1, -1]),
        "valid_task_mask": {
            "sample": torch.ones(4, dtype=torch.bool),
            "binary": torch.ones(4, dtype=torch.bool),
            "subtype": torch.tensor([False, True, True, False]),
        },
        "metadata": {"sample_id": [f"s{index}" for index in range(4)]},
    }


def test_multiclass_conformal_is_calibrated_and_serializable() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]
    )
    targets = torch.tensor([0, 1, 2, 0])
    calibrator = MulticlassConformalCalibrator.fit(probabilities, targets, alpha=0.1)
    sets = calibrator.predict_sets(probabilities)
    assert all(int(target) in values for target, values in zip(targets.tolist(), sets, strict=True))
    summary = calibrator.summarize(probabilities, targets)
    assert summary["calibration_count"] == 4
    assert summary["coverage"] == pytest.approx(1.0)


def test_subtype_metrics_use_only_valid_rows() -> None:
    result = subtype_classification_metrics(
        logits=torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        targets=torch.tensor([0, 1, 2]),
    )
    assert result["sample_count"] == 3
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_multitask_trainer_fits_and_reloads_checkpoint(tmp_path: Path) -> None:
    model = TinyMultiTask()
    config = {
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "seed": 3,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "warmup_epochs": 0,
        "early_stopping_patience": 2,
        "selection_metric": "AUROC",
        "amp": False,
        "subtype_loss_weight": 1.0,
    }
    trainer = HOSPMultiTaskTrainer(
        model,
        model_alias="hosp_bonenet.btxrd.subtype.test",
        resolved_config=config,
        checkpoint_dir=tmp_path / "run",
        device="cpu",
    )
    result = trainer.fit([_batch()], [_batch(1)])
    assert Path(result["best_checkpoint"]).is_file()
    restored = HOSPMultiTaskTrainer(
        TinyMultiTask(),
        model_alias="hosp_bonenet.btxrd.subtype.test",
        resolved_config=config,
        checkpoint_dir=tmp_path / "run",
        device="cpu",
        resume_from=Path(result["last_checkpoint"]),
    )
    assert restored.start_epoch == 2
