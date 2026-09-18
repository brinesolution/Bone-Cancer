"""Pure presentation helpers kept independent from Streamlit state."""

from __future__ import annotations

from typing import Any

import numpy as np

from bone_cancer_ai.explainability.overlays import overlay_heatmap
from bone_cancer_ai.inference.postprocess import InferenceResult

RESEARCH_DISCLAIMER = (
    "Research-use prototype only; this output is not a diagnosis and must not replace "
    "qualified clinical review."
)


def display_state(result: InferenceResult) -> dict[str, Any]:
    """Map a model result to UI text without exposing ground-truth fields."""

    return {
        "status": "abstain" if result.rejected else "prediction",
        "prediction": "bone_tumour" if result.prediction == 1 else "normal",
        "probability": result.probability,
        "confidence": result.confidence,
        "uncertainty": result.uncertainty,
        "conformal_set": result.conformal_set,
        "open_set_status": result.open_set_status,
    }


def overlay_images(result: InferenceResult) -> dict[str, np.ndarray]:
    """Return display-ready RGB image and optional heatmap overlays."""

    output = {"image": np.asarray(result.image, dtype=np.float32)}
    if result.spatial_attention is not None:
        output["spatial_attention"] = overlay_heatmap(result.image, result.spatial_attention)
    if result.gradcam is not None:
        output["gradcam"] = overlay_heatmap(result.image, result.gradcam)
    return output


def subtype_display_state(result: Any) -> dict[str, Any]:
    """Map a BTXRD result to compact, label-free application fields."""

    return {
        "binary_prediction": "bone_tumour" if result.binary_prediction == 1 else "normal",
        "binary_probability": result.binary_probability,
        "binary_confidence": result.binary_confidence,
        "binary_uncertainty": result.binary_uncertainty,
        "binary_conformal_set": result.binary_conformal_set,
        "binary_rejected": result.binary_rejected,
        "subtype_prediction": result.subtype_prediction,
        "subtype_conformal_set": result.subtype_conformal_set,
        "subtype_rejected": result.subtype_rejected,
        "subtype_status": "review" if result.subtype_rejected else "prediction",
        "top_k": result.subtype_top_k,
    }


__all__ = ["RESEARCH_DISCLAIMER", "display_state", "overlay_images", "subtype_display_state"]
