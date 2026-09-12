from __future__ import annotations

import pytest
import torch

from bone_cancer_ai.losses import (
    HOSPLoss,
    LossWeights,
    dice_bce_segmentation_loss,
)


def _predictions(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "binary_logits": torch.randn(batch_size, requires_grad=True),
        "subtype_logits": torch.randn(batch_size, 4, requires_grad=True),
        "risk_logits": torch.randn(batch_size, 2, requires_grad=True),
        "concept_logits": torch.randn(batch_size, 3, requires_grad=True),
        "concept_maps": torch.randn(batch_size, 3, 16, 16, requires_grad=True),
        "lesion_logits": torch.randn(batch_size, 1, 16, 16, requires_grad=True),
    }


def _batch(batch_size: int = 3) -> dict[str, object]:
    return {
        "binary_label": torch.tensor([0, 1, 1][:batch_size]),
        "subtype_label": torch.tensor([0, 2, 1][:batch_size]),
        "risk_label": torch.tensor([0, 1, 1][:batch_size]),
        "concept_labels": torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0]][:batch_size]
        ),
        "mask": torch.zeros(batch_size, 1, 16, 16),
        "valid_task_mask": {
            "sample": torch.ones(batch_size, dtype=torch.bool),
            "binary": torch.ones(batch_size, dtype=torch.bool),
            "subtype": torch.ones(batch_size, dtype=torch.bool),
            "risk": torch.ones(batch_size, dtype=torch.bool),
            "concepts": torch.tensor(
                [[True, True, True], [True, True, False], [True, True, True]][:batch_size]
            ),
            "mask": torch.ones(batch_size, dtype=torch.bool),
        },
    }


def test_binary_only_batch_ignores_unavailable_auxiliary_labels() -> None:
    predictions = {"binary_logits": torch.tensor([0.0, 0.5], requires_grad=True)}
    batch = {
        "binary_label": torch.tensor([0, 1]),
        "valid_task_mask": {
            "sample": torch.tensor([True, True]),
            "binary": torch.tensor([True, True]),
        },
    }
    criterion = HOSPLoss(LossWeights(binary=1.0, subtype=1.0, segmentation=1.0))

    losses = criterion(predictions, batch)
    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert losses["binary"] > 0
    assert losses["subtype"] == 0
    assert losses["segmentation"] == 0
    assert predictions["binary_logits"].grad is not None


def test_binary_and_subtype_batch_uses_only_valid_subtype_rows() -> None:
    predictions = _predictions()
    batch = _batch()
    batch["valid_task_mask"]["subtype"] = torch.tensor([True, False, True])
    criterion = HOSPLoss(LossWeights(binary=1.0, subtype=1.0))

    losses = criterion(predictions, batch)

    expected = torch.nn.functional.cross_entropy(
        predictions["subtype_logits"][[0, 2]], batch["subtype_label"][[0, 2]]
    )
    assert float(losses["subtype"].detach()) == pytest.approx(float(expected.detach()))
    assert float(losses["total"].detach()) == pytest.approx(
        float((losses["binary"] + expected).detach())
    )


def test_binary_and_mask_batch_combines_dice_and_bce() -> None:
    predictions = _predictions()
    batch = _batch()
    batch["mask"][0, :, 3:10, 4:12] = 1
    batch["valid_task_mask"]["mask"] = torch.tensor([True, False, False])
    criterion = HOSPLoss(LossWeights(binary=1.0, segmentation=1.0))

    losses = criterion(predictions, batch)

    assert torch.isfinite(losses["total"])
    assert losses["segmentation"] > 0
    assert losses["segmentation_bce"] > 0
    assert losses["segmentation_dice"] > 0


def test_full_multitask_batch_has_finite_active_terms_and_backward() -> None:
    predictions = _predictions()
    batch = _batch()
    batch["mask"][:, :, 2:12, 3:13] = 1
    criterion = HOSPLoss(
        LossWeights(
            binary=1.0,
            subtype=0.5,
            risk=0.25,
            segmentation=0.75,
            concepts=0.4,
            grounding=0.2,
            hierarchy=0.1,
        ),
        positive_subtype_indices=(0, 1, 2, 3),
    )

    losses = criterion(predictions, batch)
    losses["total"].backward()

    for name in (
        "total",
        "binary",
        "subtype",
        "risk",
        "segmentation",
        "concepts",
        "grounding",
        "hierarchy",
    ):
        assert torch.isfinite(losses[name])
        assert losses[name] > 0
    assert all(value.grad is not None for value in predictions.values())


def test_missing_every_label_returns_differentiable_finite_zero() -> None:
    predictions = _predictions(batch_size=2)
    batch = _batch(batch_size=2)
    valid = batch["valid_task_mask"]
    valid["sample"] = torch.tensor([False, False])
    valid["binary"] = torch.tensor([False, False])
    valid["subtype"] = torch.tensor([False, False])
    valid["risk"] = torch.tensor([False, False])
    valid["concepts"] = torch.zeros(2, 3, dtype=torch.bool)
    valid["mask"] = torch.tensor([False, False])
    criterion = HOSPLoss(
        LossWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        positive_subtype_indices=(0, 1, 2, 3),
    )

    losses = criterion(predictions, batch)
    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert losses["total"].item() == 0.0
    assert predictions["binary_logits"].grad is not None


def test_invalid_sentinel_values_are_ignored_outside_valid_masks() -> None:
    predictions = _predictions(batch_size=2)
    batch = _batch(batch_size=2)
    batch["binary_label"][1] = -1
    batch["subtype_label"][1] = -1
    batch["concept_labels"][1] = -1
    batch["mask"][1] = -1
    valid = batch["valid_task_mask"]
    valid["binary"][1] = False
    valid["subtype"][1] = False
    valid["concepts"][1] = False
    valid["mask"][1] = False
    criterion = HOSPLoss(
        LossWeights(binary=1.0, subtype=1.0, concepts=1.0, grounding=1.0)
    )

    losses = criterion(predictions, batch)

    assert torch.isfinite(losses["total"])


def test_inactive_full_resolution_fp16_output_cannot_poison_binary_loss() -> None:
    predictions = {
        "binary_logits": torch.zeros(1, dtype=torch.float16, requires_grad=True),
        "lesion_logits": torch.ones(
            1, 1, 384, 384, dtype=torch.float16, requires_grad=True
        ),
    }
    batch = {
        "binary_label": torch.tensor([1]),
        "valid_task_mask": {
            "sample": torch.tensor([True]),
            "binary": torch.tensor([True]),
        },
    }

    losses = HOSPLoss(LossWeights(binary=1.0))(predictions, batch)

    assert torch.isfinite(losses["total"])


def test_full_resolution_fp16_dice_reductions_are_finite() -> None:
    logits = torch.zeros(1, 1, 384, 384, dtype=torch.float16, requires_grad=True)
    targets = torch.ones_like(logits)

    loss = dice_bce_segmentation_loss(logits, targets, torch.tensor([True]))

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.bce)
    assert torch.isfinite(loss.dice)
    loss.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_valid_configured_subtype_labels_require_subtype_predictions() -> None:
    predictions = {"binary_logits": torch.zeros(2, requires_grad=True)}
    batch = {
        "binary_label": torch.tensor([0, 1]),
        "subtype_label": torch.tensor([1, 0]),
        "valid_task_mask": {
            "sample": torch.tensor([True, True]),
            "binary": torch.tensor([True, True]),
            "subtype": torch.tensor([True, True]),
        },
    }

    with pytest.raises(ValueError, match="subtype_logits.*valid subtype"):
        HOSPLoss(LossWeights(binary=1.0, subtype=1.0))(predictions, batch)
