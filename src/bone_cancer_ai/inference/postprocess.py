"""JSON-safe presentation contracts for single-image inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """One label-free research inference result and its visual explanation arrays."""

    sample_id: str
    probability: float
    prediction: int
    confidence: float
    uncertainty: float
    conformal_set: list[int]
    rejected: bool
    image: np.ndarray
    spatial_attention: np.ndarray | None
    gradcam: np.ndarray | None
    subtype: dict[str, Any] | None = None
    concepts: dict[str, Any] | None = None
    open_set_status: str = "Unavailable: no verified OOD labels"

    def to_dict(self) -> dict[str, Any]:
        """Return metadata suitable for JSON export, excluding pixel arrays."""

        payload = asdict(self)
        payload.pop("image", None)
        spatial = payload.pop("spatial_attention", None)
        gradcam = payload.pop("gradcam", None)
        payload["spatial_attention_shape"] = (
            list(spatial.shape) if isinstance(spatial, np.ndarray) else None
        )
        payload["gradcam_shape"] = (
            list(gradcam.shape) if isinstance(gradcam, np.ndarray) else None
        )
        return payload


def result_to_json(result: InferenceResult) -> str:
    """Serialize result metadata without embedding image or heatmap pixel data."""

    return json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False)


__all__ = ["InferenceResult", "result_to_json"]
