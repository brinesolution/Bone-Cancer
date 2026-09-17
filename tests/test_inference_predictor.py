from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator
from bone_cancer_ai.inference.postprocess import InferenceResult, result_to_json
from bone_cancer_ai.inference.predictor import FrozenPredictor, validate_image_array


class TinyInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.relu(self.feature_projection(images))
        return {
            "binary_logits": features.mean(dim=(1, 2, 3)),
            "lesion_logits": features[:, :1],
        }


def _predictor() -> FrozenPredictor:
    model = TinyInferenceModel().eval()
    calibrator = BinaryConformalCalibrator(alpha=0.1, qhat=0.2, calibration_count=10)
    return FrozenPredictor(
        model=model,
        calibrator=calibrator,
        device=torch.device("cpu"),
        image_size=32,
        force_grayscale=True,
    )


def test_validate_image_array_normalizes_grayscale_to_rgb() -> None:
    image = np.zeros((8, 12), dtype=np.uint8)
    validated = validate_image_array(image)
    assert validated.shape == (8, 12, 3)
    assert validated.dtype == np.uint8


@pytest.mark.parametrize(
    "image", [np.zeros((0, 5), dtype=np.uint8), np.zeros((5, 5), dtype=np.float32)]
)
def test_validate_image_array_rejects_invalid_uploads(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_image_array(image)


def test_predictor_returns_finite_binary_uncertainty_and_spatial_outputs() -> None:
    image = np.full((18, 24, 3), 180, dtype=np.uint8)
    result = _predictor().predict_array(image, sample_id="upload")

    assert isinstance(result, InferenceResult)
    assert result.sample_id == "upload"
    assert 0.0 <= result.probability <= 1.0
    assert result.prediction in {0, 1}
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.conformal_set, list)
    assert result.rejected == (len(result.conformal_set) != 1)
    assert result.spatial_attention.shape == (32, 32)
    assert result.gradcam.shape == (32, 32)
    assert np.isfinite(result.spatial_attention).all()
    assert np.isfinite(result.gradcam).all()


def test_result_json_is_metadata_only_and_json_serializable() -> None:
    result = _predictor().predict_array(np.zeros((16, 16), dtype=np.uint8))
    payload = json.loads(result_to_json(result))

    assert payload["sample_id"] == "upload"
    assert payload["spatial_attention_shape"] == [32, 32]
    assert payload["gradcam_shape"] == [32, 32]
    assert "spatial_attention" not in payload
    assert "gradcam" not in payload
    json.dumps(payload, allow_nan=False)
