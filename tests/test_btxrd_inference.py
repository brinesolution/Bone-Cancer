from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from bone_cancer_ai.evaluation.hosp_btxrd import MulticlassConformalCalibrator
from bone_cancer_ai.inference.btxrd import BtxrdSubtypePredictor
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator


class _TinySubtypeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(3, 4, kernel_size=1)
        self.binary_head = nn.Linear(4, 1)
        self.subtype_head = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_projection(images)
        pooled = features.mean(dim=(-2, -1))
        return {
            "binary_logits": self.binary_head(pooled).reshape(-1),
            "subtype_logits": self.subtype_head(pooled),
            "lesion_logits": features[:, :1],
        }


def _predictor() -> BtxrdSubtypePredictor:
    binary = BinaryConformalCalibrator.fit(
        np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0, 0, 1, 1])
    )
    subtype = MulticlassConformalCalibrator.fit(
        np.asarray(
            [
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.10, 0.10, 0.80],
                [0.70, 0.20, 0.10],
                [0.10, 0.70, 0.20],
                [0.20, 0.10, 0.70],
            ]
        ),
        np.asarray([0, 1, 2, 0, 1, 2]),
    )
    return BtxrdSubtypePredictor(
        model=_TinySubtypeModel(),
        binary_calibrator=binary,
        subtype_calibrator=subtype,
        subtype_names=("alpha", "beta", "gamma"),
        device=torch.device("cpu"),
        image_size=32,
        model_alias="test.alias",
        run_id="test-run",
        checkpoint_sha256="a" * 64,
    )


def test_btxrd_predictor_returns_binary_subtype_and_visual_evidence() -> None:
    result = _predictor().predict_array(
        np.full((17, 23, 3), 128, dtype=np.uint8), sample_id="case-1"
    )

    assert result.sample_id == "case-1"
    assert 0.0 <= result.binary_probability <= 1.0
    assert result.subtype_prediction in {"alpha", "beta", "gamma"}
    assert len(result.subtype_top_k) == 3
    assert {row["name"] for row in result.subtype_top_k} == {"alpha", "beta", "gamma"}
    assert result.image.shape == (32, 32, 3)
    assert result.spatial_attention is not None
    assert result.gradcam is not None
    assert result.spatial_attention.shape == (32, 32)
    assert result.gradcam.shape == (32, 32)
    assert result.to_dict()["image_shape"] == [32, 32, 3]
    assert "image" not in result.to_dict()


def test_btxrd_registry_loader_accepts_completed_phase_b_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    predictor = BtxrdSubtypePredictor.from_project_root(root, device="cpu")
    result = predictor.predict_array(np.zeros((12, 14, 3), dtype=np.uint8))

    assert predictor.model_alias == "hosp_bonenet.btxrd.subtype.selected"
    assert predictor.run_id.endswith("run-a8ae625f34c1")
    assert len(result.subtype_probabilities) == 9
