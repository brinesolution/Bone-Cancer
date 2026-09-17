"""Phase 9 explanation and gallery contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from bone_cancer_ai.explainability.gallery import select_gallery_cases
from bone_cancer_ai.explainability.gradcam import GradCAM
from bone_cancer_ai.explainability.overlays import overlay_heatmap


class TinySpatialClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.classifier = nn.Linear(4, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.relu(self.features(images))
        logits = self.classifier(features.mean(dim=(-2, -1))).squeeze(1)
        return {"binary_logits": logits}


def test_gradcam_returns_finite_input_aligned_maps_and_releases_hooks() -> None:
    model = TinySpatialClassifier()
    explainer = GradCAM(model, model.features)
    maps = explainer.generate(torch.randn(2, 3, 12, 10))

    assert maps.shape == (2, 12, 10)
    assert torch.isfinite(maps).all()
    assert float(maps.min()) >= 0.0
    assert float(maps.max()) <= 1.0
    with torch.inference_mode():
        model(torch.randn(1, 3, 12, 10))
    explainer.close()
    assert not explainer.active


def test_overlay_heatmap_validates_and_composites() -> None:
    image = np.full((8, 7, 3), 0.5, dtype=np.float32)
    heatmap = np.linspace(0.0, 1.0, 56, dtype=np.float32).reshape(8, 7)
    overlay = overlay_heatmap(image, heatmap, alpha=0.4)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.float32
    assert np.isfinite(overlay).all()
    assert 0.0 <= float(overlay.min()) <= float(overlay.max()) <= 1.0
    with pytest.raises(ValueError, match="shape"):
        overlay_heatmap(image, heatmap[:4], alpha=0.4)


def test_gallery_selection_covers_failures_and_uncertainty_without_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(8)],
            "target": [1, 0, 0, 1, 1, 0, 1, 0],
            "prediction": [1, 0, 1, 0, 1, 0, 1, 0],
            "probability": [0.99, 0.01, 0.95, 0.05, 0.55, 0.45, 0.52, 0.48],
            "uncertainty": [0.02, 0.02, 0.20, 0.20, 0.99, 0.99, 1.0, 1.0],
            "conformal_size": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    selected = select_gallery_cases(frame, per_category=2)

    assert {"true_positive", "true_negative", "false_positive", "false_negative"} <= set(
        selected["category"]
    )
    assert "uncertain_rejected" in set(selected["category"])
    assert not selected.duplicated(["category", "sample_id"]).any()
