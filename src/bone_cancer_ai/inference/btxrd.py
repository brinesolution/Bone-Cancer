"""Checksum-bound BTXRD subtype inference for the application layer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from bone_cancer_ai.artifacts.registry import resolve_checkpoint
from bone_cancer_ai.data.preprocessing import decode_radiograph
from bone_cancer_ai.data.transforms import RadiographTransform
from bone_cancer_ai.evaluation.calibration import binary_entropy
from bone_cancer_ai.evaluation.hosp_btxrd import MulticlassConformalCalibrator
from bone_cancer_ai.explainability.gradcam import GradCAM
from bone_cancer_ai.explainability.overlays import denormalize_image
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator
from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
from bone_cancer_ai.optimization.contracts import load_yaml, sha256_file

from .predictor import validate_image_array


@dataclass(frozen=True)
class BtxrdSubtypeResult:
    """Prediction, uncertainty, and visual evidence for one BTXRD image."""

    sample_id: str
    binary_probability: float
    binary_prediction: int
    binary_confidence: float
    binary_uncertainty: float
    binary_conformal_set: list[int]
    binary_rejected: bool
    subtype_probabilities: dict[str, float]
    subtype_prediction: str
    subtype_top_k: list[dict[str, Any]]
    subtype_conformal_set: list[str]
    subtype_rejected: bool
    image: np.ndarray
    spatial_attention: np.ndarray | None
    gradcam: np.ndarray | None
    model_alias: str
    run_id: str
    checkpoint_sha256: str
    evaluation_protocol: str = "btxrd_fresh_calibration_and_test"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata while keeping image arrays out of downloads."""

        return {
            "sample_id": self.sample_id,
            "binary_probability": self.binary_probability,
            "binary_prediction": self.binary_prediction,
            "binary_confidence": self.binary_confidence,
            "binary_uncertainty": self.binary_uncertainty,
            "binary_conformal_set": list(self.binary_conformal_set),
            "binary_rejected": self.binary_rejected,
            "subtype_probabilities": dict(self.subtype_probabilities),
            "subtype_prediction": self.subtype_prediction,
            "subtype_top_k": list(self.subtype_top_k),
            "subtype_conformal_set": list(self.subtype_conformal_set),
            "subtype_rejected": self.subtype_rejected,
            "model_alias": self.model_alias,
            "run_id": self.run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evaluation_protocol": self.evaluation_protocol,
            "image_shape": list(self.image.shape),
            "spatial_attention_shape": (
                list(self.spatial_attention.shape)
                if self.spatial_attention is not None
                else None
            ),
            "gradcam_shape": list(self.gradcam.shape) if self.gradcam is not None else None,
        }


def btxrd_result_to_json(result: BtxrdSubtypeResult) -> str:
    """Serialize one subtype result without embedding image or heatmap arrays."""

    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


class BtxrdSubtypePredictor:
    """Run the completed Phase B checkpoint with masked subtype supervision."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        binary_calibrator: BinaryConformalCalibrator,
        subtype_calibrator: MulticlassConformalCalibrator,
        subtype_names: Sequence[str],
        device: torch.device | str,
        image_size: int = 384,
        force_grayscale: bool = True,
        model_alias: str = "",
        run_id: str = "",
        checkpoint_sha256: str = "",
    ) -> None:
        names = tuple(str(name) for name in subtype_names)
        if len(names) != subtype_calibrator.num_classes or len(names) < 2:
            raise ValueError("subtype_names must match the subtype calibrator class count")
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("subtype_names must be unique and non-empty")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.model = model.eval()
        self.binary_calibrator = binary_calibrator
        self.subtype_calibrator = subtype_calibrator
        self.subtype_names = names
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.force_grayscale = bool(force_grayscale)
        self.model_alias = str(model_alias)
        self.run_id = str(run_id)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.transform = RadiographTransform(
            self.image_size, train=False, seed=42, normalize=True
        )

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        *,
        alias: str = "hosp_bonenet.btxrd.subtype.selected",
        config_path: Path | None = None,
        device: torch.device | str | None = None,
    ) -> BtxrdSubtypePredictor:
        """Load only the completed, registry-bound BTXRD artifact."""

        root = Path(project_root).resolve()
        config_file = (
            Path(config_path)
            if config_path is not None
            else root / "configs" / "hosp_btxrd_subtype.yaml"
        )
        config_file = config_file if config_file.is_absolute() else root / config_file
        config = load_yaml(config_file)
        if not str(config.get("status", "")).startswith("phase_b_"):
            raise ValueError("BTXRD predictor config contract is invalid")
        model_config = config.get("model")
        if not isinstance(model_config, Mapping) or model_config.get("name") != "hosp_bonenet":
            raise ValueError("BTXRD model config is invalid")
        schema_file = Path(str(config["label_schema"]))
        schema_file = schema_file if schema_file.is_absolute() else root / schema_file
        schema = load_yaml(schema_file)
        subtype_schema = schema.get("subtype")
        if not isinstance(subtype_schema, Mapping) or not subtype_schema.get("enabled"):
            raise ValueError("BTXRD subtype schema is disabled")
        raw_names = subtype_schema.get("classes")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
            raise ValueError("BTXRD subtype schema classes are invalid")

        record = resolve_checkpoint(root, alias)
        if record.phase != "phase_b" or record.model_family != "hosp_bonenet":
            raise ValueError("requested checkpoint is not the completed BTXRD HOSP artifact")
        if record.test_status.get("status") != "complete":
            raise ValueError("BTXRD checkpoint has no completed fresh evaluation")
        run_dir = Path(record.source_run)
        run_dir = run_dir if run_dir.is_absolute() else root / run_dir
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.is_file():
            raise FileNotFoundError(metrics_file)
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        status_file = run_dir / "status.json"
        status = (
            json.loads(status_file.read_text(encoding="utf-8"))
            if status_file.is_file()
            else {}
        )
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(status, Mapping)
            or status.get("status") != "complete"
        ):
            raise ValueError("BTXRD run status is not complete")
        if metrics.get("protocol") != "btxrd_fresh_calibration_and_test":
            raise ValueError("BTXRD run protocol is not fresh calibration and test")
        if metrics.get("configuration_fingerprint") != record.configuration_fingerprint:
            raise ValueError("BTXRD run fingerprint differs from registry")
        evaluation = metrics.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("BTXRD evaluation metrics are missing")
        binary_state = evaluation.get("binary_conformal")
        subtype_state = evaluation.get("subtype_calibrator")
        if not isinstance(binary_state, Mapping) or not isinstance(subtype_state, Mapping):
            raise ValueError("BTXRD conformal calibration state is missing")
        binary = BinaryConformalCalibrator(
            alpha=float(binary_state["alpha"]),
            qhat=float(binary_state["qhat"]),
            calibration_count=int(binary_state["calibration_count"]),
        )
        subtype = MulticlassConformalCalibrator.from_state_dict(dict(subtype_state))
        data_file = Path(str(config["data_config"]))
        data_file = data_file if data_file.is_absolute() else root / data_file
        data_config = load_yaml(data_file)
        requested = torch.device(
            str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configured CUDA subtype inference device is unavailable")
        resolved_model = dict(model_config)
        # All parameters come from the checksum-bound checkpoint; avoid a network weight fetch.
        resolved_model["pretrained"] = False
        model = create_hosp_bonenet(resolved_model, schema)
        checkpoint_file = record.absolute_checkpoint_path(root)
        if sha256_file(checkpoint_file) != record.checkpoint_sha256:
            raise ValueError("BTXRD checkpoint checksum changed while loading")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("BTXRD checkpoint does not contain model_state")
        model.load_state_dict(state, strict=True)
        return cls(
            model=model.to(requested),
            binary_calibrator=binary,
            subtype_calibrator=subtype,
            subtype_names=tuple(str(name) for name in raw_names),
            device=requested,
            image_size=int(data_config.get("image_size", 384)),
            force_grayscale=bool(data_config.get("preprocessing", {}).get("force_grayscale", True)),
            model_alias=record.alias,
            run_id=record.source_run,
            checkpoint_sha256=record.checkpoint_sha256,
        )

    def _attention(self, output: Mapping[str, Any]) -> np.ndarray | None:
        logits = output.get("lesion_logits")
        if not isinstance(logits, torch.Tensor):
            return None
        attention = F.interpolate(
            logits[:1].sigmoid(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return attention.detach().cpu().numpy().astype(np.float32)

    def predict_array(self, image: np.ndarray, *, sample_id: str = "upload") -> BtxrdSubtypeResult:
        """Predict binary status and subtype while preserving visual evidence."""

        pixels = validate_image_array(image)
        if self.force_grayscale:
            grayscale = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            pixels = np.repeat(grayscale[..., None], 3, axis=2)
        transformed = self.transform(image=pixels)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        if not isinstance(output, Mapping):
            raise RuntimeError("BTXRD model returned an invalid output")
        binary_logits = output.get("binary_logits")
        subtype_logits = output.get("subtype_logits")
        if (
            not isinstance(binary_logits, torch.Tensor)
            or binary_logits.numel() != 1
            or not isinstance(subtype_logits, torch.Tensor)
            or subtype_logits.shape != (1, len(self.subtype_names))
        ):
            raise RuntimeError("BTXRD model did not return configured binary and subtype logits")
        probability = float(torch.sigmoid(binary_logits.reshape(-1)[0]).item())
        binary_prediction = int(probability >= 0.5)
        binary_set = self.binary_calibrator.predict_sets(np.asarray([probability]))[0]
        subtype_probabilities_array = torch.softmax(subtype_logits[0].to(torch.float64), dim=0)
        subtype_probabilities = subtype_probabilities_array.cpu().numpy().astype(np.float64)
        subtype_indices = np.argsort(-subtype_probabilities, kind="stable")
        top_k = [
            {
                "index": int(index),
                "name": self.subtype_names[int(index)],
                "probability": float(subtype_probabilities[int(index)]),
            }
            for index in subtype_indices[: min(5, len(subtype_indices))]
        ]
        subtype_set_indices = self.subtype_calibrator.predict_sets(
            subtype_probabilities[None, :]
        )[0]
        subtype_set = [self.subtype_names[index] for index in subtype_set_indices]
        with GradCAM(self.model, self.model.feature_projection) as explainer:
            gradcam = explainer.generate(
                tensor,
                target_classes=torch.tensor([binary_prediction], device=self.device),
            )[0].detach().cpu().numpy().astype(np.float32)
        return BtxrdSubtypeResult(
            sample_id=str(sample_id),
            binary_probability=probability,
            binary_prediction=binary_prediction,
            binary_confidence=max(probability, 1.0 - probability),
            binary_uncertainty=float(binary_entropy(np.asarray([probability]))[0]),
            binary_conformal_set=binary_set,
            binary_rejected=len(binary_set) != 1,
            subtype_probabilities={
                name: float(subtype_probabilities[index])
                for index, name in enumerate(self.subtype_names)
            },
            subtype_prediction=self.subtype_names[int(subtype_indices[0])],
            subtype_top_k=top_k,
            subtype_conformal_set=subtype_set,
            subtype_rejected=len(subtype_set) != 1,
            image=denormalize_image(tensor[0]),
            spatial_attention=self._attention(output),
            gradcam=gradcam,
            model_alias=self.model_alias,
            run_id=self.run_id,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def predict_path(self, path: Path) -> BtxrdSubtypeResult:
        """Decode one local image and run subtype-aware inference."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        return self.predict_array(
            decode_radiograph(source, force_grayscale=self.force_grayscale),
            sample_id=source.name,
        )


__all__ = ["BtxrdSubtypePredictor", "BtxrdSubtypeResult", "btxrd_result_to_json"]
