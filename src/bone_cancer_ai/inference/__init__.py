"""Frozen-model inference and presentation helpers."""

from .postprocess import InferenceResult, result_to_json
from .predictor import FrozenPredictor, validate_image_array

__all__ = [
    "FrozenPredictor",
    "InferenceResult",
    "result_to_json",
    "validate_image_array",
]
