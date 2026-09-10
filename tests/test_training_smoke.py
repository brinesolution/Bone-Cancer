"""CPU-only contract tests for the shared Phase 5 binary trainer."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


class TwoFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images)[:, 0]


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


def _vector_batch(
    values: list[list[float]],
    labels: list[int],
    ids: list[str],
    *,
    sample_valid: list[bool] | None = None,
) -> dict[str, Any]:
    size = len(values)
    return {
        "image": torch.tensor(values, dtype=torch.float32),
        "binary_label": torch.tensor(labels, dtype=torch.long),
        "valid_task_mask": {
            "sample": torch.tensor(sample_valid or [True] * size),
            "binary": torch.tensor(sample_valid or [True] * size),
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


def _fit_frame(labels: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "assigned_split": ["fit_train"] * len(labels),
            "is_eligible": [True] * len(labels),
            "binary_label": labels,
        }
    )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (_fit_frame([]), "empty"),
        (_fit_frame([0, 0]), "both classes"),
        (pd.DataFrame({"binary_label": [0, 1]}), "missing columns"),
        (
            pd.DataFrame(
                {"binary_label": [0, 1], "assigned_split": ["fit_train", "fit_train"]}
            ),
            "missing columns",
        ),
        (pd.DataFrame({"binary_label": [0, 1], "is_eligible": [True, True]}), "missing columns"),
        (_fit_frame([0, 1, None]), "present numeric"),
        (_fit_frame([0, 1, 2]), "only 0 and 1"),
        (
            pd.DataFrame(
                {
                    "binary_label": [0, 1],
                    "assigned_split": ["fit_train", "val"],
                    "is_eligible": [True, True],
                }
            ),
            "only fit_train",
        ),
        (
            pd.DataFrame(
                {
                    "binary_label": [0, 1],
                    "assigned_split": ["fit_train", "fit_train"],
                    "is_eligible": [True, False],
                }
            ),
            "eligible fit rows",
        ),
    ],
)
def test_class_weight_rejects_inputs_that_could_corrupt_fit_weighting(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
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


def test_cosine_scheduler_without_warmup_starts_at_configured_rate() -> None:
    optimizer = build_optimizer(TinyBinaryModel(), learning_rate=0.1, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, total_epochs=4, warmup_epochs=0)

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates == pytest.approx([0.1, 0.075, 0.025, 0.0])


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


def _checkpoint_target_state() -> tuple[
    TinyBinaryModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
    EarlyStopping,
]:
    model = TinyBinaryModel(-4.0)
    optimizer = build_optimizer(model, learning_rate=0.02, weight_decay=0.0)
    model.weight.grad = torch.tensor(0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = build_scheduler(optimizer, total_epochs=3, warmup_epochs=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    stopping = EarlyStopping(patience=2)
    stopping.step(0.4, epoch=1)
    return model, optimizer, scheduler, scaler, stopping


def _training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    stopping: EarlyStopping,
) -> dict[str, Any]:
    return deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "stopping": stopping.state_dict(),
        }
    )


def _assert_nested_state_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _saved_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    source = TinyBinaryModel(1.25)
    optimizer = build_optimizer(source, learning_rate=0.01, weight_decay=0.0)
    source.weight.grad = torch.tensor(0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = build_scheduler(optimizer, total_epochs=3, warmup_epochs=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    stopping = EarlyStopping(patience=2)
    stopping.step(0.75, epoch=1)
    config = {"epochs": 3, "seed": 17}
    checkpoint = tmp_path / "valid.pt"
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
        resolved_config=config,
        history=[{"epoch": 1, "val_AUROC": 0.75}],
    )
    return checkpoint, config


def test_checkpoint_config_is_validated_before_any_state_mutation(tmp_path: Path) -> None:
    checkpoint, _config = _saved_checkpoint(tmp_path)
    model, optimizer, scheduler, scaler, stopping = _checkpoint_target_state()
    before = _training_state(model, optimizer, scheduler, scaler, stopping)

    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=stopping,
            expected_model_alias="densenet121",
            expected_resolved_config={"epochs": 4, "seed": 17},
            map_location="cpu",
        )

    _assert_nested_state_equal(
        _training_state(model, optimizer, scheduler, scaler, stopping), before
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "selection_history",
        "optimizer_parameters",
        "optimizer_moment",
        "scheduler_rates",
    ],
)
def test_malformed_checkpoint_leaves_all_supplied_state_unchanged(
    tmp_path: Path, corruption: str
) -> None:
    checkpoint, config = _saved_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if corruption == "selection_history":
        payload["selection_value"] = 0.5
    elif corruption == "optimizer_parameters":
        payload["optimizer_state"]["param_groups"][0]["params"] = []
    elif corruption == "optimizer_moment":
        parameter_id = payload["optimizer_state"]["param_groups"][0]["params"][0]
        payload["optimizer_state"]["state"][parameter_id]["exp_avg"] = torch.zeros(2)
    else:
        payload["scheduler_state"]["base_lrs"] = []
    malformed = tmp_path / f"{corruption}.pt"
    torch.save(payload, malformed)
    model, optimizer, scheduler, scaler, stopping = _checkpoint_target_state()
    before = _training_state(model, optimizer, scheduler, scaler, stopping)

    with pytest.raises(ValueError):
        load_checkpoint(
            malformed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=stopping,
            expected_model_alias="densenet121",
            expected_resolved_config=config,
            map_location="cpu",
        )

    _assert_nested_state_equal(
        _training_state(model, optimizer, scheduler, scaler, stopping), before
    )


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


def test_accumulation_matches_one_batch_with_unequal_valid_counts_and_partial_tail(
    tmp_path: Path,
) -> None:
    accumulated_batches = [
        _vector_batch([[1.0, 0.0], [1.0, 0.0]], [0, 0], ["a", "b"]),
        _vector_batch(
            [[-1.5, 1.0], [99.0, 99.0]],
            [0, -1],
            ["tail", "rejected"],
            sample_valid=[True, False],
        ),
    ]
    combined_batch = [
        _vector_batch(
            [[1.0, 0.0], [1.0, 0.0], [-1.5, 1.0]],
            [0, 0, 0],
            ["a", "b", "tail"],
        )
    ]
    validation = [
        _vector_batch(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [0, 0, 1, 1],
            ["v0", "v1", "v2", "v3"],
        )
    ]
    accumulated_config = _config(epochs=1, accumulation=3)
    accumulated_config["gradient_clip_norm"] = 100.0
    combined_config = _config(epochs=1, accumulation=1)
    combined_config["gradient_clip_norm"] = 100.0
    accumulated = BinaryTrainer(
        TwoFeatureModel(),
        model_alias="densenet121",
        resolved_config=accumulated_config,
        checkpoint_dir=tmp_path / "accumulated",
        pos_weight=1.0,
        device="cpu",
    )
    combined = BinaryTrainer(
        TwoFeatureModel(),
        model_alias="densenet121",
        resolved_config=combined_config,
        checkpoint_dir=tmp_path / "combined",
        pos_weight=1.0,
        device="cpu",
    )

    accumulated_result = accumulated.fit(accumulated_batches, validation)
    combined_result = combined.fit(combined_batch, validation)

    assert accumulated_result["history"][0]["optimizer_steps"] == 1
    assert accumulated_result["history"][0]["train_loss"] == pytest.approx(
        combined_result["history"][0]["train_loss"]
    )
    assert torch.equal(accumulated.model.linear.weight, combined.model.linear.weight)
    accumulated_moment = next(iter(accumulated.optimizer.state.values()))["exp_avg"]
    combined_moment = next(iter(combined.optimizer.state.values()))["exp_avg"]
    assert torch.equal(accumulated_moment, combined_moment)


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


class SeededJitter:
    def __init__(self) -> None:
        self.generator = random.Random(0)

    def set_seed(self, seed: int) -> None:
        self.generator.seed(seed)

    def __call__(self, value: float) -> float:
        return value + 0.1 * (self.generator.random() - 0.5)


class StochasticMappingDataset(Dataset[dict[str, Any]]):
    def __init__(self) -> None:
        self.values = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        self.labels = [0, 0, 0, 1, 1, 1]
        self.transform = SeededJitter()

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        jittered = self.transform(self.values[index])
        jittered += 0.05 * (random.random() - 0.5)
        jittered += 0.05 * (float(np.random.random()) - 0.5)
        return {
            "image": torch.tensor([jittered], dtype=torch.float32),
            "binary_label": torch.tensor(self.labels[index], dtype=torch.long),
            "valid_task_mask": {
                "sample": torch.tensor(True),
                "binary": torch.tensor(True),
            },
            "metadata": {"sample_id": f"train-{index}"},
        }


class InterruptingLoader(Iterable[dict[str, Any]]):
    def __init__(self, loader: DataLoader[dict[str, Any]], complete_epochs: int) -> None:
        self.loader = loader
        self.complete_epochs = complete_epochs
        self.iterations = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.iterations += 1
        if self.iterations > self.complete_epochs:
            raise RuntimeError("synthetic stochastic interruption")
        return iter(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


class PersistentWorkerLoader(Iterable[dict[str, Any]]):
    num_workers = 1
    persistent_workers = True

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(_train_batches())


def _stochastic_model() -> nn.Module:
    torch.manual_seed(1234)
    return nn.Sequential(
        nn.Linear(1, 4),
        nn.Tanh(),
        nn.Dropout(p=0.4),
        nn.Linear(4, 1),
    )


def _shuffled_loader() -> DataLoader[dict[str, Any]]:
    return DataLoader(
        StochasticMappingDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(9999),
    )


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


def test_resume_matches_uninterrupted_stochastic_model_and_shuffled_loader(
    tmp_path: Path,
) -> None:
    config = _config(epochs=4, patience=8)
    uninterrupted = BinaryTrainer(
        _stochastic_model(),
        model_alias="densenet121",
        resolved_config=config,
        checkpoint_dir=tmp_path / "uninterrupted-stochastic",
        pos_weight=1.0,
        device="cpu",
    )
    uninterrupted_result = uninterrupted.fit(_shuffled_loader(), _val_batches())

    run_dir = tmp_path / "resumed-stochastic"
    interrupted = BinaryTrainer(
        _stochastic_model(),
        model_alias="densenet121",
        resolved_config=config,
        checkpoint_dir=run_dir,
        pos_weight=1.0,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="stochastic interruption"):
        interrupted.fit(InterruptingLoader(_shuffled_loader(), 2), _val_batches())

    resumed = BinaryTrainer(
        _stochastic_model(),
        model_alias="densenet121",
        resolved_config=config,
        checkpoint_dir=run_dir,
        pos_weight=1.0,
        device="cpu",
        resume_from=run_dir / "last.pt",
    )
    resumed_result = resumed.fit(_shuffled_loader(), _val_batches())

    assert resumed_result["history"] == uninterrupted_result["history"]
    for resumed_parameter, uninterrupted_parameter in zip(
        resumed.model.parameters(), uninterrupted.model.parameters(), strict=True
    ):
        assert torch.equal(resumed_parameter, uninterrupted_parameter)


def test_fit_rejects_persistent_workers_that_cannot_be_reseeded_each_epoch(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path, epochs=1)

    with pytest.raises(ValueError, match="persistent_workers"):
        trainer.fit(PersistentWorkerLoader(), _val_batches())


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


def test_fit_uses_only_train_and_validation_while_predict_is_a_separate_operation(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    trainer = _trainer(tmp_path, epochs=1)
    train = SpyLoader("train", _train_batches(), events)
    val = SpyLoader("val", _val_batches(), events)
    test = SpyLoader("test", _val_batches(), events)

    trainer.fit(train, val)

    assert "test" not in events
    fit_events = list(events)

    prediction = trainer.predict(test)

    assert events == [*fit_events, "test"]
    assert prediction["sample_ids"] == ["v0", "v1", "v2", "v3"]
