"""Strict LabelMe geometry conversion and derived spatial artifacts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

_BOUNDARY_TOLERANCE = 1e-6
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    if any(part.casefold() == "dataset" for part in resolved.parts):
        raise ValueError("Derived output must not be written under a dataset directory")
    return resolved


def _validated_points(
    shape: dict[str, Any], *, width: int, height: int
) -> tuple[str, np.ndarray]:
    shape_type = shape.get("shape_type")
    if shape_type not in {"rectangle", "polygon"}:
        raise ValueError(f"Unsupported shape_type: {shape_type!r}")
    try:
        points = np.asarray(shape["points"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Shape points must be numeric [x, y] pairs") from exc
    required = 2 if shape_type == "rectangle" else 3
    invalid_rectangle_count = shape_type == "rectangle" and len(points) != required
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or invalid_rectangle_count
        or len(points) < required
    ):
        raise ValueError(f"{shape_type} points have invalid shape")
    if not np.isfinite(points).all():
        raise ValueError("Shape coordinates must be finite")
    # LabelMe may serialize arithmetic noise at an image edge. Values within
    # 1e-6 pixel are clipped; anything farther outside fails closed.
    if (
        (points[:, 0] < -_BOUNDARY_TOLERANCE).any()
        or (points[:, 0] > width + _BOUNDARY_TOLERANCE).any()
        or (points[:, 1] < -_BOUNDARY_TOLERANCE).any()
        or (points[:, 1] > height + _BOUNDARY_TOLERANCE).any()
    ):
        raise ValueError("Shape coordinates are outside image bounds")
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)
    if shape_type == "rectangle":
        if points[0, 0] == points[1, 0] or points[0, 1] == points[1, 1]:
            raise ValueError("Rectangle geometry is degenerate")
    else:
        x, y = points[:, 0], points[:, 1]
        twice_area = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        if twice_area <= _BOUNDARY_TOLERANCE:
            raise ValueError("Polygon geometry is degenerate")
    return shape_type, points


def convert_labelme(
    payload: dict[str, Any], image_path: Path, allowed_labels: list[str]
) -> dict[str, Any]:
    """Validate and convert LabelMe rectangles and polygons at source resolution."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    allowed = set(allowed_labels)
    if not allowed or any(not isinstance(item, str) or not item for item in allowed):
        raise ValueError("allowed_labels must contain non-empty class labels")
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        orientation = int(image.getexif().get(274, 1))
        if orientation != 1:
            raise ValueError(f"EXIF orientation {orientation} is unsupported")
        width, height = image.size

    if Path(str(payload.get("imagePath", ""))).name != image_path.name:
        raise ValueError("LabelMe imagePath does not match expected image basename")
    if payload.get("imageWidth") != width or payload.get("imageHeight") != height:
        raise ValueError("LabelMe dimensions do not match actual image dimensions")
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("LabelMe shapes must be a list")

    polygons: list[np.ndarray] = []
    labels: list[str] = []
    boxes: list[list[float]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            raise ValueError("Each LabelMe shape must be an object")
        label = shape.get("label")
        if label not in allowed:
            raise ValueError(f"Invalid class label: {label!r}")
        labels.append(label)
        shape_type, points = _validated_points(shape, width=width, height=height)
        if shape_type == "polygon":
            polygons.append(points)
        else:
            x_values, y_values = points[:, 0], points[:, 1]
            boxes.append(
                [
                    float(x_values.min()),
                    float(y_values.min()),
                    float(x_values.max()),
                    float(y_values.max()),
                ]
            )

    mask = None
    if polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        for points in polygons:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return {"mask": mask, "boxes": boxes, "width": width, "height": height, "labels": labels}


def save_spatial(result: dict[str, Any], output_root: Path, sample_id: str) -> dict[str, Any]:
    """Atomically persist a binary mask, when present, and box metadata."""
    if not _SAFE_ID.fullmatch(sample_id):
        raise ValueError(f"Unsafe sample_id: {sample_id!r}")
    width, height = int(result["width"]), int(result["height"])
    if width < 1 or height < 1:
        raise ValueError("Spatial dimensions must be positive")
    output_root = _safe_output_path(output_root)
    mask_dir, box_dir = output_root / "masks", output_root / "boxes"
    mask_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)

    mask_path: Path | None = None
    mask = result.get("mask")
    expected_mask_path = mask_dir / f"{sample_id}.png"
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != (height, width):
            raise ValueError("Mask dimensions do not match the spatial result")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must be binary")
        mask_path = expected_mask_path
        temporary = mask_path.with_name(f".{mask_path.name}.{uuid4().hex}.tmp.png")
        try:
            Image.fromarray(mask.astype(np.uint8), mode="L").save(temporary, format="PNG")
            os.replace(temporary, mask_path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        expected_mask_path.unlink(missing_ok=True)

    box_path = box_dir / f"{sample_id}.json"
    box_payload = {
        "sample_id": sample_id,
        "width": width,
        "height": height,
        "boxes": result.get("boxes", []),
        "labels": result.get("labels", []),
    }
    temporary_json = box_path.with_name(f".{box_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_json.write_text(
            json.dumps(box_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_json, box_path)
    finally:
        temporary_json.unlink(missing_ok=True)
    return {"mask_path": mask_path, "box_path": box_path, "overlay_path": None}


def render_overlay(
    image_path: Path, result: dict[str, Any], destination: Path
) -> Path:
    """Render a source-resolution QA overlay without altering the source image."""
    image_path = Path(image_path).resolve(strict=True)
    destination = _safe_output_path(destination)
    if destination == image_path:
        raise ValueError("Overlay destination must not overwrite the source image")
    with Image.open(image_path) as opened:
        if int(opened.getexif().get(274, 1)) != 1:
            raise ValueError("EXIF orientation is unsupported for overlays")
        image = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    height, width = image.shape[:2]
    if (width, height) != (int(result["width"]), int(result["height"])):
        raise ValueError("Overlay image dimensions do not match spatial result")
    mask = result.get("mask")
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != (height, width):
            raise ValueError("Mask dimensions do not match overlay image")
        selected = mask.astype(bool)
        image[selected] = np.rint(image[selected] * 0.55 + np.array([115, 0, 0])).astype(
            np.uint8
        )
    for box in result.get("boxes", []):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 215, 0), thickness=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.png")
    try:
        Image.fromarray(image, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
