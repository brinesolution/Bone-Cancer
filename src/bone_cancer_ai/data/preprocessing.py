"""Safe radiograph decoding and aspect-ratio-preserving spatial preparation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

CACHE_FORMAT_VERSION = "radiograph-png-v1-exif-grayscale-opencv-aspect-pad"


def decode_radiograph(path: Path, *, force_grayscale: bool = True) -> np.ndarray:
    """Decode one source image in memory and return uint8 RGB pixels."""
    path = Path(path)
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        converted = (
            oriented.convert("L").convert("RGB")
            if force_grayscale
            else oriented.convert("RGB")
        )
        converted.load()
        return np.asarray(converted, dtype=np.uint8).copy()


def load_binary_mask(path: Path, *, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load a binary mask without interpolation or source modification."""
    with Image.open(Path(path)) as opened:
        oriented = ImageOps.exif_transpose(opened).convert("L")
        oriented.load()
        mask = (np.asarray(oriented, dtype=np.uint8) > 0).astype(np.uint8)
    if expected_size is not None and mask.shape != expected_size:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {expected_size}")
    return mask


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy array")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [H,W,3] or [H,W]")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image dimensions must be positive")
    return image


def resize_and_pad(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    size: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Resize by one scale factor, then pad centrally to a square canvas."""
    image = _validate_image(image)
    if size < 1:
        raise ValueError("size must be positive")
    if not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be in [0,255]")
    original_height, original_width = image.shape[:2]
    if mask is not None:
        if mask.ndim != 2 or mask.shape != (original_height, original_width):
            raise ValueError("mask must have shape [H,W] matching the image")
        mask = (mask > 0).astype(np.uint8)
    scale = min(size / original_height, size / original_width)
    resized_height = max(1, min(size, int(round(original_height * scale))))
    resized_width = max(1, min(size, int(round(original_width * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    resized_mask = (
        cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        if mask is not None
        else None
    )
    pad_height = size - resized_height
    pad_width = size - resized_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    padded_mask = (
        cv2.copyMakeBorder(
            resized_mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        if resized_mask is not None
        else None
    )
    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return padded, padded_mask, metadata


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_preprocess_fingerprint(
    source_sha256: str,
    *,
    image_size: int,
    force_grayscale: bool,
    pad_value: int,
) -> str:
    """Identify deterministic pre-augmentation pixels for one audited source image."""
    config = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "image_size": int(image_size),
        "force_grayscale": bool(force_grayscale),
        "pad_value": int(pad_value),
    }
    payload = f"{source_sha256}|{json.dumps(config, sort_keys=True, separators=(',', ':'))}"
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_manifest_images(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    image_size: int,
    force_grayscale: bool,
    pad_value: int,
    limit: int | None = None,
    prior_index: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build optional deterministic PNG cache entries keyed only by opaque sample IDs."""
    required = {"sample_id", "absolute_path", "sha256", "is_eligible"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Cache manifest missing columns: {sorted(missing)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows = manifest.loc[manifest["is_eligible"].fillna(False).astype(bool)].copy()
    rows = rows.sort_values("sample_id")
    if limit is not None:
        rows = rows.head(limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "image_size": int(image_size),
        "force_grayscale": bool(force_grayscale),
        "pad_value": int(pad_value),
    }
    prior_by_sample: dict[str, pd.Series] = {}
    if prior_index is not None and not prior_index.empty:
        if "sample_id" not in prior_index or prior_index["sample_id"].duplicated().any():
            raise ValueError("Prior cache index requires unique sample_id values")
        prior_by_sample = {
            str(row["sample_id"]): row for _, row in prior_index.iterrows()
        }
    records = []
    reused = 0
    built = 0
    rebuilt = 0
    for row in rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        if Path(sample_id).name != sample_id or not sample_id:
            raise ValueError(f"Unsafe sample_id for cache path: {sample_id!r}")
        source_path = Path(str(row.absolute_path)).resolve()
        destination = output_dir / f"{sample_id}.png"
        status = "cached"
        error_type = ""
        cache_sha256 = ""
        fingerprint = cache_preprocess_fingerprint(
            str(row.sha256),
            image_size=image_size,
            force_grayscale=force_grayscale,
            pad_value=pad_value,
        )
        original_height = 0
        original_width = 0
        action = "built"
        try:
            if _file_sha256(source_path) != str(row.sha256):
                raise RuntimeError("SourceHashMismatch")
            prior = prior_by_sample.get(sample_id)
            prior_height = int(prior.get("original_height", 0)) if prior is not None else 0
            prior_width = int(prior.get("original_width", 0)) if prior is not None else 0
            prior_matches = (
                prior is not None
                and str(prior.get("cache_path", "")) == str(destination.resolve())
                and str(prior.get("assigned_split", ""))
                == str(getattr(row, "assigned_split", ""))
                and str(prior.get("source_sha256", "")) == str(row.sha256)
                and str(prior.get("preprocess_fingerprint", "")) == fingerprint
                and str(prior.get("status", "")) == "cached"
                and destination.is_file()
                and str(prior.get("cache_sha256", "")) == _file_sha256(destination)
                and prior_height > 0
                and prior_width > 0
            )
            if prior_matches:
                cache_sha256 = str(prior["cache_sha256"])
                original_height = prior_height
                original_width = prior_width
                action = "reused"
                reused += 1
            else:
                action = "rebuilt" if prior is not None or destination.exists() else "built"
                image = decode_radiograph(source_path, force_grayscale=force_grayscale)
                original_height, original_width = image.shape[:2]
                prepared, _, _ = resize_and_pad(
                    image, None, size=image_size, pad_value=pad_value
                )
                temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.png")
                try:
                    Image.fromarray(prepared, mode="RGB").save(temporary, format="PNG")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                cache_sha256 = _file_sha256(destination)
                if action == "rebuilt":
                    rebuilt += 1
                else:
                    built += 1
        except Exception as exc:
            destination.unlink(missing_ok=True)
            status = "failed"
            error_type = str(exc) if str(exc) == "SourceHashMismatch" else type(exc).__name__
            action = "failed"
        records.append(
            {
                "sample_id": sample_id,
                "assigned_split": str(getattr(row, "assigned_split", "")),
                "cache_path": str(destination.resolve()),
                "source_sha256": str(row.sha256),
                "cache_sha256": cache_sha256,
                "preprocess_fingerprint": fingerprint,
                "cache_format_version": CACHE_FORMAT_VERSION,
                "image_size": int(image_size),
                "force_grayscale": bool(force_grayscale),
                "pad_value": int(pad_value),
                "original_height": int(original_height),
                "original_width": int(original_width),
                "status": status,
                "error_type": error_type,
                "action": action,
            }
        )
    index = pd.DataFrame(records)
    summary = {
        "requested": len(rows),
        "cached": int(index["status"].eq("cached").sum()) if not index.empty else 0,
        "failed": int(index["status"].eq("failed").sum()) if not index.empty else 0,
        "built": built,
        "rebuilt": rebuilt,
        "reused": reused,
        "limited": limit is not None,
        "config": config_payload,
    }
    return index, summary
