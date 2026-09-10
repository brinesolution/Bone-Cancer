"""Safe staging and import helpers for LabelMe."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from bone_cancer_ai.annotations.spatial import convert_labelme

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_equivalent(left: Path, right: Path) -> bool:
    """Return whether two files have exactly equal bytes, established by SHA-256."""
    left, right = Path(left), Path(right)
    return left.stat().st_size == right.stat().st_size and _sha256(left) == _sha256(right)


def stage_labelme(
    image_path: Path,
    sample_id: str,
    annotation_root: Path,
    expected_sha256: str,
) -> dict[str, Path]:
    """Copy a hash-verified image into an opaque, source-safe LabelMe workspace."""
    if not _SAFE_ID.fullmatch(sample_id):
        raise ValueError(f"Unsafe sample_id: {sample_id!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal hash")
    source = Path(image_path).resolve(strict=True)
    actual_hash = _sha256(source)
    if actual_hash != expected_sha256.lower():
        raise ValueError("Source image hash does not match expected hash")
    with Image.open(source) as image:
        orientation = int(image.getexif().get(274, 1))
        if orientation != 1:
            raise ValueError(f"EXIF orientation {orientation} is unsupported")
        width, height = image.size

    annotation_root = Path(annotation_root).resolve()
    if any(part.casefold() == "dataset" for part in annotation_root.parts):
        raise ValueError("LabelMe staging must not be written under a dataset directory")
    stage_dir = annotation_root / "labelme" / sample_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_image = stage_dir / f"{sample_id}{source.suffix.lower()}"
    json_path = stage_dir / f"{sample_id}.json"
    binding_path = stage_dir / f"{sample_id}.binding.json"
    if any(path.exists() for path in (staged_image, json_path, binding_path)):
        raise FileExistsError(f"LabelMe staging already exists for {sample_id!r}")

    temporary_image = staged_image.with_name(f".{staged_image.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary_image)
        if _sha256(temporary_image) != actual_hash:
            raise OSError("Staged image copy failed byte verification")
        os.replace(temporary_image, staged_image)
    finally:
        temporary_image.unlink(missing_ok=True)

    labelme_payload = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [],
        "imagePath": staged_image.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    binding = {
        "binding_version": 1,
        "sample_id": sample_id,
        "source_image_path": str(source),
        "source_sha256": actual_hash,
        "staged_image": staged_image.name,
        "labelme_json": json_path.name,
    }
    try:
        json_path.write_text(json.dumps(labelme_payload, indent=2) + "\n", encoding="utf-8")
        binding_path.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    except Exception:
        staged_image.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        binding_path.unlink(missing_ok=True)
        raise
    return {
        "image_path": staged_image,
        "json_path": json_path,
        "binding_path": binding_path,
    }


def import_staged(binding_path: Path, allowed_labels: list[str]) -> dict[str, Any]:
    """Validate a staging binding and convert its edited LabelMe JSON."""
    binding_path = Path(binding_path).resolve(strict=True)
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid LabelMe binding JSON") from exc
    required = {
        "binding_version",
        "sample_id",
        "source_image_path",
        "source_sha256",
        "staged_image",
        "labelme_json",
    }
    if not isinstance(binding, dict) or required - binding.keys():
        raise ValueError("LabelMe binding is incomplete")
    sample_id = binding["sample_id"]
    invalid_sample_id = not isinstance(sample_id, str) or not _SAFE_ID.fullmatch(sample_id)
    if binding["binding_version"] != 1 or invalid_sample_id:
        raise ValueError("LabelMe binding metadata is invalid")
    if binding_path.parent.name != sample_id:
        raise ValueError("LabelMe binding directory does not match sample_id")

    def bound_file(name: Any, expected_name: str | None = None) -> Path:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("LabelMe binding contains an unsafe staged path")
        if expected_name is not None and name != expected_name:
            raise ValueError("LabelMe binding filename does not match sample_id")
        return binding_path.parent / name

    staged_image = bound_file(binding["staged_image"])
    json_path = bound_file(binding["labelme_json"], f"{sample_id}.json")
    source = Path(binding["source_image_path"]).resolve(strict=True)
    expected_hash = binding["source_sha256"]
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("LabelMe binding source hash is invalid")
    if _sha256(source) != expected_hash:
        raise ValueError("Bound source image has changed")
    if not staged_image.is_file() or not verify_equivalent(source, staged_image):
        raise ValueError("Staged image no longer matches bound source image")
    try:
        json_bytes = json_path.read_bytes()
        payload = json.loads(json_bytes.decode("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid or missing staged LabelMe JSON") from exc
    result = convert_labelme(payload, staged_image, allowed_labels)
    result.update(
        {
            "sample_id": sample_id,
            "source_image_path": source,
            "source_sha256": expected_hash,
            "staged_image_path": staged_image,
            "labelme_json_path": json_path,
            "labelme_json_bytes": json_bytes,
            "binding_path": binding_path,
        }
    )
    return result
