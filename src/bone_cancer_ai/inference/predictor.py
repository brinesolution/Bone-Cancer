"""Checksum-bound, label-free single-image HOSP-BoneNet inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from bone_cancer_ai.data.preprocessing import decode_radiograph
from bone_cancer_ai.data.transforms import RadiographTransform
from bone_cancer_ai.evaluation.calibration import binary_entropy
from bone_cancer_ai.explainability.gradcam import GradCAM
from bone_cancer_ai.explainability.overlays import denormalize_image
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator
from bone_cancer_ai.optimization.contracts import load_yaml
from bone_cancer_ai.optimization.frozen import load_frozen_phase8_checkpoint

from .postprocess import InferenceResult


def validate_image_array(image: np.ndarray) -> np.ndarray:
    """Validate an uploaded uint8 image and return three-channel RGB pixels."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("uploaded image must be a uint8 NumPy array")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("uploaded image must have shape [H,W] or [H,W,3]")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("uploaded image dimensions must be positive")
    return np.ascontiguousarray(image)


class FrozenPredictor:
    """Run frozen HOSP-BoneNet inference without requiring labels."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        calibrator: BinaryConformalCalibrator,
        device: torch.device,
        image_size: int = 384,
        force_grayscale: bool = True,
    ) -> None:
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.model = model.eval()
        self.calibrator = calibrator
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.force_grayscale = bool(force_grayscale)
        self.transform = RadiographTransform(
            self.image_size,
            train=False,
            seed=42,
            normalize=True,
        )

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        config_path: Path | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> FrozenPredictor:
        """Load the selected checkpoint and canonical conformal artifact."""

        root = Path(project_root).resolve()
        config_file = (
            Path(config_path)
            if config_path is not None
            else root / "configs" / "inference.yaml"
        )
        config_file = config_file if config_file.is_absolute() else root / config_file
        config = load_yaml(Path(config_file))
        if config.get("evaluation_status") != "phase_9_complete":
            raise ValueError("inference requires a completed Phase 9 evaluation")
        requested = torch.device(str(device or config.get("device", "cuda")))
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configured CUDA inference device is unavailable")
        checkpoint = str(config["checkpoint_alias"])
        model, _, _ = load_frozen_phase8_checkpoint(root, checkpoint, device=requested)
        calibration_path = Path(str(config["conformal_calibration"]))
        calibration_path = (
            calibration_path if calibration_path.is_absolute() else root / calibration_path
        )
        calibration = (
            load_yaml(calibration_path)
            if calibration_path.suffix in {".yaml", ".yml"}
            else None
        )
        if calibration is None:
            import json

            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibrator = BinaryConformalCalibrator(
            alpha=float(calibration["alpha"]),
            qhat=float(calibration["qhat"]),
            calibration_count=int(calibration["calibration_count"]),
        )
        data_path = Path(str(config["data_config"]))
        data_config = load_yaml(data_path if data_path.is_absolute() else root / data_path)
        return cls(
            model=model,
            calibrator=calibrator,
            device=requested,
            image_size=int(data_config.get("image_size", 384)),
            force_grayscale=bool(data_config.get("preprocessing", {}).get("force_grayscale", True)),
        )

    def _attention(self, output: Mapping[str, Any]) -> np.ndarray | None:
        logits = output.get("lesion_logits")
        if not isinstance(logits, torch.Tensor):
            return None
        attention = logits[:1].sigmoid()
        attention = F.interpolate(
            attention, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )[0, 0]
        return attention.detach().cpu().numpy().astype(np.float32)

    def predict_array(self, image: np.ndarray, *, sample_id: str = "upload") -> InferenceResult:
        """Preprocess and predict one uploaded image, including explanation maps."""

        pixels = validate_image_array(image)
        if self.force_grayscale:
            grayscale = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            pixels = np.repeat(grayscale[..., None], 3, axis=2)
        transformed = self.transform(image=pixels)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        logits = output.get("binary_logits")
        if not isinstance(logits, torch.Tensor) or logits.numel() != 1:
            raise RuntimeError("frozen model did not return one binary logit")
        probability = float(torch.sigmoid(logits.reshape(-1)[0]).item())
        prediction = int(probability >= 0.5)
        sets = self.calibrator.predict_sets(np.asarray([probability], dtype=np.float64))[0]
        with GradCAM(self.model, self.model.feature_projection) as explainer:
            gradcam = explainer.generate(
                tensor, target_classes=torch.tensor([prediction], device=self.device)
            )[0].detach().cpu().numpy().astype(np.float32)
        display_image = denormalize_image(tensor[0])
        return InferenceResult(
            sample_id=str(sample_id),
            probability=probability,
            prediction=prediction,
            confidence=max(probability, 1.0 - probability),
            uncertainty=float(binary_entropy(np.asarray([probability]))[0]),
            conformal_set=sets,
            rejected=len(sets) != 1,
            image=display_image,
            spatial_attention=self._attention(output),
            gradcam=gradcam,
            subtype=None,
            concepts=None,
        )

    def predict_path(self, path: Path) -> InferenceResult:
        """Decode one local image in memory and predict it without modifying the file."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        return self.predict_array(
            decode_radiograph(source, force_grayscale=self.force_grayscale),
            sample_id=source.name,
        )


__all__ = ["FrozenPredictor", "validate_image_array"]
