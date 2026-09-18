from __future__ import annotations

import numpy as np

from app.ui_helpers import RESEARCH_DISCLAIMER, display_state, overlay_images, subtype_display_state
from bone_cancer_ai.inference.postprocess import InferenceResult


def _result(*, rejected: bool) -> InferenceResult:
    return InferenceResult(
        sample_id="upload",
        probability=0.8,
        prediction=1,
        confidence=0.8,
        uncertainty=0.7,
        conformal_set=[] if rejected else [1],
        rejected=rejected,
        image=np.zeros((8, 8, 3), dtype=np.float32),
        spatial_attention=np.zeros((8, 8), dtype=np.float32),
        gradcam=np.zeros((8, 8), dtype=np.float32),
    )


def test_display_state_exposes_abstention_without_ground_truth() -> None:
    state = display_state(_result(rejected=True))
    assert state["status"] == "abstain"
    assert state["prediction"] == "bone_tumour"
    assert "target" not in state
    assert "ground" not in " ".join(state).lower()


def test_overlay_images_returns_three_channel_finite_pixels() -> None:
    result = _result(rejected=False)
    images = overlay_images(result)
    assert set(images) == {"image", "spatial_attention", "gradcam"}
    assert images["spatial_attention"].shape == (8, 8, 3)
    assert all(np.isfinite(value).all() for value in images.values())


def test_disclaimer_is_research_use_only() -> None:
    assert "research" in RESEARCH_DISCLAIMER.lower()
    assert "not a diagnosis" in RESEARCH_DISCLAIMER.lower()


def test_subtype_display_state_exposes_top_class_and_ambiguity() -> None:
    state = subtype_display_state(
        type(
            "Result",
            (),
            {
                "binary_prediction": 1,
                "binary_probability": 0.91,
                "binary_confidence": 0.91,
                "binary_uncertainty": 0.2,
                "binary_conformal_set": [1],
                "binary_rejected": False,
                "subtype_prediction": "osteosarcoma",
                "subtype_conformal_set": ["osteosarcoma", "other mt"],
                "subtype_rejected": True,
                "subtype_top_k": [
                    {"name": "osteosarcoma", "probability": 0.7},
                    {"name": "other mt", "probability": 0.2},
                ],
            },
        )()
    )
    assert state["binary_prediction"] == "bone_tumour"
    assert state["subtype_prediction"] == "osteosarcoma"
    assert state["subtype_status"] == "review"
    assert len(state["top_k"]) == 2
