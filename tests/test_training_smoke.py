"""CPU-only contract tests for the shared Phase 5 binary trainer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

from bone_cancer_ai.training.callbacks import EarlyStopping
from bone_cancer_ai.training.checkpointing import load_checkpoint, save_checkpoint
from bone_cancer_ai.training.optimizer import build_optimizer, class_weight_from_frame
from bone_cancer_ai.training.scheduler import build_scheduler
from bone_cancer_ai.training.trainer import BinaryTrainer


class TinyBinaryModel(nn.Module):
    """One-parameter binary model whose ranking stays stable during smoke training."""

    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, 0] * self.weight


def _batch(
    values: list[float],
    labels: list[int],
    ids: list[str],
    *,
    sample_valid: list[bool] | None = None,
    binary_valid: list[bool] | None = None,
) -> dict[str, Any]:
    size = len(values)
    return {
        "image": torch.tensor(values, dtype=torch.float32).view(size, 1),
        "binary_label": torch.tensor(labels, dtype=torch.long),
        "valid_task_mask": {
            "sample": torch.tensor(sample_valid or [True] * size),
            "binary": torch.tensor(binary_valid or [True] * size),
        },
        "metadata": {"sample_id": ids},
    }


def _config(*, epochs: int = 3, accumulation: int = 1, patience: int = 8) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "warmup_epochs": 1,
        "amp": True,
        "gradient_accumulation_steps": accumulation,
        "gradient_clip_norm": 0.05,
        "early_stopping_patience": patience,
        "seed": 17,
        "selection_metric": "AUROC",
    }


def _trainer(
    directory: Path,
    *,
    epochs: int = 3,
    accumulation: int = 1,
    patience: int = 8,
    weight: float = 0.5,
) -> BinaryTrainer:
    return BinaryTrainer(
        TinyBinaryModel(weight),
        model_alias="densenet121",
        resolved_config=_config(
            epochs=epochs,
            accumulation=accumulation,
            patience=patience,
        ),
        checkpoint_dir=directory,
        pos_weight=torch.tensor(1.0),
        device="cpu",
    )


def _train_batches() -> list[dict[str, Any]]:
    return [
        _batch([-2.0, 2.0], [0, 1], ["fit-a", "fit-b"]),
        _batch([-1.0, 1.0], [0, 1], ["fit-c", "fit-d"]),
        _batch([50.0, -50.0], [-1, -1], ["bad-a", "bad-b"], sample_valid=[False, False]),
        _batch([-3.0, 3.0], [0, 1], ["fit-e", "fit-f"]),
    ]


def _val_batches() -> list[dict[str, Any]]:
    return [_batch([-2.5, -0.5, 0.5, 2.5], [0, 0, 1, 1], ["v0", "v1", "v2", "v3"])]


def test_class_weight_uses_only_well_formed_fit_binary_labels() -> None:
    frame = pd.DataFrame(
        {
            "assigned_split": ["fit_train"] * 6,
            "is_eligible": [True] * 6,
            "binary_label": [0, 0, 0, 0, 1, 1],
        }
    )

    weight = class_weight_from_frame(frame)

    assert weight.dtype == torch.float32
    assert weight.ndim == 0
    assert weight.item() == 2.0


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"binary_label": []}),
        pd.DataFrame({"binary_label": [0, 0]}),
        pd.DataFrame({"binary_label": [0, 1]}),
        pd.DataFrame(
            {"binary_label": [0, 1], "assigned_split": ["fit_train", "fit_train"]}
        ),
        pd.DataFrame({"binary_label": [0, 1], "is_eligible": [True, True]}),
        pd.DataFrame({"binary_label": [0, 1, None]}),
        pd.DataFrame({"binary_label": [0, 1, 2]}),
        pd.DataFrame({"binary_label": [0, 1], "assigned_split": ["fit_train", "val"]}),
        pd.DataFrame({"binary_label": [0, 1], "is_eligible": [True, False]}),
    ],
)
def test_class_weight_rejects_inputs_that_could_corrupt_fit_weighting(
    frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        class_weight_from_frame(frame)


def test_optimizer_and_scheduler_use_adamw_linear_warmup_then_cosine() -> None:
    model = TinyBinaryModel()
    optimizer = build_optimizer(model, learning_rate=0.1, weight_decay=0.02)
    scheduler = build_scheduler(optimizer, total_epochs=4, warmup_epochs=2)

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == 0.02
    assert learning_rates == pytest.approx([0.05, 0.1, 0.05, 0.0])


def test_early_stopping_is_strict_and_resume_safe() -> None:
    stopping = EarlyStopping(patience=2)

    assert stopping.step(0.6, epoch=1) is False
    assert stopping.step(0.6, epoch=2) is False
    state = stopping.state_dict()

    restored = EarlyStopping(patience=2)
    restored.load_state_dict(state)

    assert restored.best_value == 0.6
    assert restored.best_epoch == 1
    assert restored.step(0.59, epoch=3) is True
    with pytest.raises(ValueError):
        restored.step(float("nan"), epoch=4)


def test_checkpoint_alias_is_validated_before_model_state_mutation(tmp_path: Path) -> None:
    source = TinyBinaryModel(1.25)
    optimizer = build_optimizer(source, learning_rate=0.01, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, total_epochs=3, warmup_epochs=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    stopping = EarlyStopping(patience=2)
    stopping.step(0.75, epoch=1)
    checkpoint = tmp_path / "nested" / "last.pt"

    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=stopping,
        model_alias="densenet121",
        epoch=1,
        selection_metric="AUROC",
        selection_value=0.75,
        resolved_config={"epochs": 3, "seed": 17},
        history=[{"epoch": 1, "val_AUROC": 0.75}],
    )
    target = TinyBinaryModel(-4.0)

    with pytest.raises(ValueError, match="alias"):
        load_checkpoint(
            checkpoint,
            model=target,
            expected_model_alias="swin_tiny",
            map_location="cpu",
        )

    assert target.weight.item() == -4.0
    metadata = load_checkpoint(
        checkpoint,
        model=target,
        expected_model_alias="densenet121",
        map_location="cpu",
    )
    assert target.weight.item() == pytest.approx(1.25)
    assert metadata["epoch"] == 1
    assert metadata["selection_metric"] == "AUROC"
    assert metadata["selection_value"] == 0.75
    json.dumps(metadata["history"], allow_nan=False)
    assert not list(checkpoint.parent.glob(f".{checkpoint.name}.*.tmp"))


def test_fit_accumulates_partial_window_clips_updates_and_writes_best_last(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path, epochs=5, accumulation=2, patience=2)
    initial = trainer.model.weight.detach().clone()

    result = trainer.fit(_train_batches(), _val_batches())

    assert result["best_epoch"] == 1
    assert result["best_AUROC"] == 1.0
    assert len(result["history"]) == 3
    assert result["history"][0]["optimizer_steps"] == 2
    assert result["history"][0]["train_valid_samples"] == 6
    assert result["history"][0]["train_rejected_samples"] == 2
    assert all(record["gradient_norm"] <= 0.050001 for record in result["history"])
    assert torch.isfinite(trainer.model.weight).all()
    assert not torch.equal(initial, trainer.model.weight.detach())
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    best = torch.load(tmp_path / "best.pt", map_location="cpu")
    last = torch.load(tmp_path / "last.pt", map_location="cpu")
    assert best["epoch"] == 1
    assert last["epoch"] == 3
    json.dumps(result, allow_nan=False)


def test_fit_applies_bce_positive_class_weight(tmp_path: Path) -> None:
    loader = [_batch([0.0, 0.0], [0, 1], ["negative", "positive"])]
    unweighted = BinaryTrainer(
        TinyBinaryModel(0.0),
        model_alias="densenet121",
        resolved_config=_config(epochs=1),
        checkpoint_dir=tmp_path / "unweighted",
        pos_weight=1.0,
        device="cpu",
    )
    weighted = BinaryTrainer(
        TinyBinaryModel(0.0),
        model_alias="densenet121",
        resolved_config=_config(epochs=1),
        checkpoint_dir=tmp_path / "weighted",
        pos_weight=3.0,
        device="cpu",
    )

    unweighted_result = unweighted.fit(loader, _val_batches())
    weighted_result = weighted.fit(loader, _val_batches())

    assert unweighted_result["history"][0]["train_loss"] == pytest.approx(0.693147, rel=1e-5)
    assert weighted_result["history"][0]["train_loss"] == pytest.approx(1.386294, rel=1e-5)


def test_fit_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    first = _trainer(tmp_path / "first", epochs=2)
    second = _trainer(tmp_path / "second", epochs=2)

    first_result = first.fit(_train_batches(), _val_batches())
    second_result = second.fit(_train_batches(), _val_batches())

    assert first_result["history"] == second_result["history"]
    assert torch.equal(first.model.weight.detach(), second.model.weight.detach())


class InterruptAfter(Iterable[dict[str, Any]]):
    def __init__(self, batches: list[dict[str, Any]], complete_epochs: int) -> None:
        self.batches = batches
        self.complete_epochs = complete_epochs
        self.iterations = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.iterations += 1
        if self.iterations > self.complete_epochs:
            raise RuntimeError("synthetic interruption")
        return iter(self.batches)


def test_resume_restores_history_scheduler_optimizer_and_early_stopping(tmp_path: Path) -> None:
    uninterrupted = _trainer(tmp_path / "uninterrupted", epochs=4, patience=8)
    uninterrupted_result = uninterrupted.fit(_train_batches(), _val_batches())

    run_dir = tmp_path / "resumed"
    interrupted = _trainer(run_dir, epochs=4, patience=8)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        interrupted.fit(InterruptAfter(_train_batches(), complete_epochs=2), _val_batches())

    resumed = BinaryTrainer(
        TinyBinaryModel(-9.0),
        model_alias="densenet121",
        resolved_config=_config(epochs=4, patience=8),
        checkpoint_dir=run_dir,
        pos_weight=1.0,
        device="cpu",
        resume_from=run_dir / "last.pt",
    )
    resumed_result = resumed.fit(_train_batches(), _val_batches())

    assert [row["epoch"] for row in resumed_result["history"]] == [1, 2, 3, 4]
    assert resumed_result["history"] == uninterrupted_result["history"]
    assert torch.equal(resumed.model.weight.detach(), uninterrupted.model.weight.detach())


@pytest.mark.parametrize("operation", ["fit", "predict"])
def test_missing_valid_binary_samples_fail_instead_of_borrowing_targets(
    tmp_path: Path, operation: str
) -> None:
    trainer = _trainer(tmp_path, epochs=1)
    invalid = [
        _batch(
            [1.0, -1.0],
            [-1, -1],
            ["invalid-1", "invalid-2"],
            binary_valid=[False, False],
        )
    ]

    with pytest.raises(ValueError, match="no valid binary samples"):
        if operation == "fit":
            trainer.fit(invalid, _val_batches())
        else:
            trainer.predict(invalid)


def test_predict_filters_invalid_rows_preserves_order_and_exact_module_modes(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.BatchNorm1d(1), TinyBinaryModel())
    model.train()
    model[0].eval()
    trainer = BinaryTrainer(
        model,
        model_alias="densenet121",
        resolved_config=_config(epochs=1),
        checkpoint_dir=tmp_path,
        pos_weight=1.0,
        device="cpu",
    )
    loader = [
        _batch(
            [-2.0, 99.0, 2.0],
            [0, -1, 1],
            ["opaque-a", "opaque-rejected", "opaque-b"],
            sample_valid=[True, False, True],
        )
    ]

    prediction = trainer.predict(loader)

    assert prediction["logits"].device.type == "cpu"
    assert prediction["targets"].tolist() == [0, 1]
    assert prediction["sample_ids"] == ["opaque-a", "opaque-b"]
    assert prediction["rejected_count"] == 1
    assert torch.isfinite(prediction["logits"]).all()
    assert model.training is True
    assert model[0].training is False


class SpyLoader(Iterable[dict[str, Any]]):
    def __init__(self, name: str, batches: list[dict[str, Any]], events: list[str]) -> None:
        self.name = name
        self.batches = batches
        self.events = events

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.events.append(self.name)
        return iter(self.batches)


def test_fit_never_reads_test_and_test_prediction_follows_best_checkpoint_load(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    trainer = _trainer(tmp_path, epochs=1)
    train = SpyLoader("train", _train_batches(), events)
    val = SpyLoader("val", _val_batches(), events)
    test = SpyLoader("test", _val_batches(), events)

    trainer.fit(train, val)

    assert "test" not in events
    load_checkpoint(
        tmp_path / "best.pt",
        model=trainer.model,
        expected_model_alias="densenet121",
        map_location="cpu",
    )
    events.append("best-loaded")
    trainer.predict(test)
    assert events.index("best-loaded") < events.index("test")
