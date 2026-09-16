"""Manifest-only PyTorch dataset with explicit partial-task validity masks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from bone_cancer_ai.data.preprocessing import decode_radiograph, load_binary_mask
from bone_cancer_ai.data.transforms import RadiographTransform

LOGGER = logging.getLogger("bone_cancer_ai.data")


def _class_map(section: dict[str, Any]) -> dict[Any, int]:
    classes = section.get("classes", [])
    if isinstance(classes, dict):
        result = {}
        for key in classes:
            try:
                normalized = int(key)
            except (TypeError, ValueError):
                normalized = str(key)
            encoded = int(key) if isinstance(key, int) or str(key).isdigit() else len(result)
            result[normalized] = encoded
        return result
    return {str(value): index for index, value in enumerate(classes)}


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list | tuple | dict):
        return False
    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing or str(value).strip() == ""
    return False


class BoneCancerDataset(Dataset):
    """Load only manifest records; never parse labels from filenames or paths."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: RadiographTransform,
        label_schema: dict[str, Any],
        *,
        force_grayscale: bool = True,
    ) -> None:
        required = {
            "sample_id",
            "absolute_path",
            "assigned_split",
            "binary_label",
            "is_eligible",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset manifest missing columns: {sorted(missing)}")
        if not frame["is_eligible"].fillna(False).astype(bool).all():
            raise ValueError("Dataset manifest contains excluded records")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("sample_id values must be present and unique")
        self.frame = frame.sort_values("sample_id").reset_index(drop=True).copy()
        self.transform = transform
        self.schema = label_schema
        self.force_grayscale = bool(force_grayscale)
        self.binary_map = _class_map(label_schema.get("binary", {}))
        self.subtype_map = _class_map(label_schema.get("subtype", {}))
        self.risk_map = _class_map(label_schema.get("risk", {}))
        self.concept_names = list(label_schema.get("concepts", {}).get("names", []))

    def __len__(self) -> int:
        return len(self.frame)

    def _encode(self, row: pd.Series, task: str, value: Any) -> tuple[int, bool]:
        section = self.schema.get(task, {})
        if not bool(section.get("enabled", False)) or _missing(value):
            return -1, False
        mapping = {
            "binary": self.binary_map,
            "subtype": self.subtype_map,
            "risk": self.risk_map,
        }[task]
        normalized = int(value) if task == "binary" else str(value)
        if normalized not in mapping:
            return -1, False
        return int(mapping[normalized]), True

    def _concepts(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(self.concept_names)
        values = torch.full((count,), -1.0, dtype=torch.float32)
        valid = torch.zeros((count,), dtype=torch.bool)
        if not bool(self.schema.get("concepts", {}).get("enabled", False)) or count == 0:
            return values, valid
        raw = row.get("concept_labels")
        if _missing(raw):
            return values, valid
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return values, valid
        if not isinstance(parsed, list) or len(parsed) != count:
            return values, valid
        for index, value in enumerate(parsed):
            if value in (0, 1, 0.0, 1.0, False, True):
                values[index] = float(value)
                valid[index] = True
        return values, valid

    def _labels(self, row: pd.Series) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        binary, binary_valid = self._encode(row, "binary", row.get("binary_label"))
        subtype, subtype_valid = self._encode(row, "subtype", row.get("subtype_label"))
        risk, risk_valid = self._encode(row, "risk", row.get("risk_label"))
        concepts, concept_valid = self._concepts(row)
        labels = {
            "binary_label": torch.tensor(binary, dtype=torch.long),
            "subtype_label": torch.tensor(subtype, dtype=torch.long),
            "risk_label": torch.tensor(risk, dtype=torch.long),
            "concept_labels": concepts,
        }
        valid = {
            "binary": torch.tensor(binary_valid, dtype=torch.bool),
            "subtype": torch.tensor(subtype_valid, dtype=torch.bool),
            "risk": torch.tensor(risk_valid, dtype=torch.bool),
            "concepts": concept_valid,
        }
        return labels, valid

    def _invalid_item(
        self, row: pd.Series, labels: dict[str, torch.Tensor], error: Exception
    ) -> dict[str, Any]:
        size = self.transform.image_size
        valid = {
            "sample": torch.tensor(False),
            "binary": torch.tensor(False),
            "subtype": torch.tensor(False),
            "risk": torch.tensor(False),
            "concepts": torch.zeros((len(self.concept_names),), dtype=torch.bool),
            "mask": torch.tensor(False),
        }
        LOGGER.warning(
            "Runtime image decode failed for %s: %s",
            row["sample_id"],
            type(error).__name__,
        )
        return {
            "image": torch.zeros((3, size, size), dtype=torch.float32),
            **labels,
            "mask": torch.zeros((1, size, size), dtype=torch.float32),
            "valid_task_mask": valid,
            "metadata": {
                "sample_id": str(row["sample_id"]),
                "assigned_split": str(row["assigned_split"]),
                "original_height": int(row.get("height", 0) or 0),
                "original_width": int(row.get("width", 0) or 0),
                "load_error": type(error).__name__,
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        labels, valid = self._labels(row)
        cache_path = row.get("cache_path")
        use_cache = not _missing(cache_path)
        try:
            image_path = Path(str(cache_path if use_cache else row["absolute_path"]))
            image = decode_radiograph(
                image_path, force_grayscale=False if use_cache else self.force_grayscale
            )
        except Exception as exc:
            if use_cache:
                raise RuntimeError(
                    f"Required cache entry failed for sample {row['sample_id']}"
                ) from exc
            return self._invalid_item(row, labels, exc)
        mask = None
        mask_valid = False
        mask_path = row.get("mask_path")
        if (
            bool(self.schema.get("localization", {}).get("masks_verified", False))
            and not _missing(mask_path)
        ):
            try:
                mask = load_binary_mask(Path(str(mask_path)), expected_size=image.shape[:2])
                mask_valid = True
            except Exception as exc:
                LOGGER.warning(
                    "Runtime mask decode failed for %s: %s",
                    row["sample_id"],
                    type(exc).__name__,
                )
        transformed = self.transform(image=image, mask=mask)
        valid.update(
            {
                "sample": torch.tensor(True),
                "mask": torch.tensor(mask_valid),
            }
        )
        return {
            "image": transformed["image"],
            **labels,
            "mask": transformed["mask"],
            "valid_task_mask": valid,
            "metadata": {
                "sample_id": str(row["sample_id"]),
                "assigned_split": str(row["assigned_split"]),
                "original_height": int(row.get("height", image.shape[0]) or image.shape[0]),
                "original_width": int(row.get("width", image.shape[1]) or image.shape[1]),
                "load_error": "",
            },
        }
