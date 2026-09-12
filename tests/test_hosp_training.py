from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from bone_cancer_ai.training.checkpointing import load_checkpoint
from bone_cancer_ai.training.trainer import BinaryTrainer


class TinyHOSPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.ones_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(images).squeeze(1)
        return {
            "binary_logits": logits,
            "lesion_logits": logits[:, None, None, None].expand(-1, 1, 4, 4),
        }


def _batch(values: list[float], labels: list[int], prefix: str) -> dict[str, Any]:
    size = len(values)
    return {
        "image": torch.tensor(values, dtype=torch.float32).unsqueeze(1),
        "binary_label": torch.tensor(labels, dtype=torch.long),
        "valid_task_mask": {
            "sample": torch.ones(size, dtype=torch.bool),
            "binary": torch.ones(size, dtype=torch.bool),
        },
        "metadata": {"sample_id": [f"{prefix}-{index}" for index in range(size)]},
    }


def _config() -> dict[str, object]:
    return {
        "epochs": 2,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 10.0,
        "seed": 42,
        "selection_metric": "AUROC",
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "warmup_epochs": 0,
        "early_stopping_patience": 8,
        "amp": False,
        "stage_name": "A_binary_global_local",
        "monitor_extended": True,
    }


def test_shared_trainer_handles_hosp_mapping_and_records_phase7_monitors(
    tmp_path: Path, capsys: Any,
) -> None:
    torch.manual_seed(7)
    model = TinyHOSPModel()
    config = _config()
    trainer = BinaryTrainer(
        model,
        model_alias="hosp_bonenet",
        resolved_config=config,
        checkpoint_dir=tmp_path,
        pos_weight=1.0,
        device="cpu",
    )
    train = [
        _batch([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1], "train")
    ]
    val = [_batch([-3.0, -0.5, 0.5, 3.0], [0, 0, 1, 1], "val")]

    result = trainer.fit(train, val)
    live_output = capsys.readouterr().out

    assert [row["stage"] for row in result["history"]] == [
        "A_binary_global_local",
        "A_binary_global_local",
    ]
    for row in result["history"]:
        assert row["train_binary_loss"] == row["train_loss"]
        assert row["val_binary_loss"] == row["val_loss"]
        assert 0 <= row["val_AUPRC"] <= 1
        assert 0 <= row["val_F1"] <= 1
        assert row["epoch_time_seconds"] >= 0
        assert row["peak_cuda_memory_bytes"] == 0
    assert live_output.count("phase7_epoch ") == 2
    assert "stage=A_binary_global_local" in live_output
    assert "val_AUROC=" in live_output
    assert "checkpoint_decision=best" in live_output
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()

    load_checkpoint(
        tmp_path / "best.pt",
        model=model,
        expected_model_alias="hosp_bonenet",
        map_location="cpu",
        expected_resolved_config=config,
        expected_selection_metric="AUROC",
    )
    expected = trainer.predict(val)["logits"]
    restored = TinyHOSPModel()
    load_checkpoint(
        tmp_path / "best.pt",
        model=restored,
        expected_model_alias="hosp_bonenet",
        map_location="cpu",
        expected_resolved_config=config,
        expected_selection_metric="AUROC",
    )
    reloaded = BinaryTrainer(
        restored,
        model_alias="hosp_bonenet",
        resolved_config=config,
        checkpoint_dir=tmp_path / "unused",
        pos_weight=1.0,
        device="cpu",
    ).predict(val)["logits"]
    assert torch.equal(reloaded, expected)


def test_extended_monitor_prints_undefined_threshold_metric_without_failing(
    tmp_path: Path, capsys: Any
) -> None:
    model = TinyHOSPModel()
    model.linear.weight.data.zero_()
    model.linear.bias.data.fill_(-100.0)
    config = {**_config(), "epochs": 1}
    trainer = BinaryTrainer(
        model,
        model_alias="hosp_bonenet",
        resolved_config=config,
        checkpoint_dir=tmp_path,
        pos_weight=1.0,
        device="cpu",
    )
    batch = _batch([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1], "constant")

    result = trainer.fit([batch], [batch])

    assert result["history"][0]["val_F1"] is None
    assert "val_F1=undefined" in capsys.readouterr().out
