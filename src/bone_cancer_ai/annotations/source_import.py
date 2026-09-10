"""Import mechanically validated BTXRD LabelMe annotations by exact image identity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePath
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image

from bone_cancer_ai.annotations.context import load_annotation_context
from bone_cancer_ai.annotations.spatial import convert_labelme, render_overlay, save_spatial
from bone_cancer_ai.utils.config import load_config

_MASTER_COLUMNS = [
    "sample_id",
    "image_id",
    "source_sha256",
    "source_json_sha256",
    "mask_path",
    "box_path",
    "width",
    "height",
    "source_json_path",
    "source_json_aliases",
    "assigned_split",
    "subtype_label",
    "risk_label",
    "status",
    "provenance",
    "reviewer",
    "clinical_verification",
]
_ISSUE_COLUMNS = ["source_json_path", "source_image_path", "error_code", "error"]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _data_config(config_path: Path, annotation_config: dict[str, Any]) -> dict[str, Any]:
    path = Path(annotation_config["data_config"])
    return load_config(path if path.is_absolute() else config_path.parent / path)


def _source_root(context: Any, data: dict[str, Any]) -> Path:
    configured = Path(data["provenance"]["btxrd_root"])
    root = (
        configured if configured.is_absolute() else context.paths.project_root / configured
    ).resolve()
    if not root.is_relative_to(context.paths.project_root):
        raise ValueError("BTXRD source root escapes the project root")
    return root


def _safe_image_name(payload: dict[str, Any], json_path: Path) -> str:
    value = payload.get("imagePath")
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("unsafe_image_basename")
    if PurePath(value).stem != json_path.stem:
        raise ValueError("json_image_basename_mismatch")
    return value


def _write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _issue(path: Path, image: Path | None, code: str, error: str) -> dict[str, str]:
    return {
        "source_json_path": str(path.resolve()),
        "source_image_path": str(image.resolve()) if image else "",
        "error_code": code,
        "error": error,
    }


def _spatial_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    mask = result.get("mask")
    return (
        int(result["width"]),
        int(result["height"]),
        tuple(tuple(box) for box in result["boxes"]),
        tuple(result["labels"]),
        None
        if mask is None
        else (mask.shape, mask.dtype.str, hashlib.sha256(mask.tobytes()).digest()),
    )


def import_source_annotations(config_path: Path) -> dict[str, Any]:
    """Convert BTXRD JSON only for eligible Main images with identical SHA-256 bytes."""
    config_path = Path(config_path).resolve()
    context = load_annotation_context(config_path)
    data = _data_config(config_path, context.config)
    source_root = _source_root(context, data)
    json_root, image_root = source_root / "Annotations", source_root / "images"
    if not json_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError("BTXRD Annotations and images directories are required")

    generation_root = context.root / "source_imports" / uuid4().hex
    generation_root.mkdir(parents=True, exist_ok=False)
    allowed_labels = list(context.schema.get("subtype", {}).get("classes", []))
    allowed_labels.extend(context.config.get("spatial_labels", ["lesion"]))
    allowed_labels = list(dict.fromkeys(allowed_labels))
    by_hash = {
        digest: group.to_dict("records")
        for digest, group in context.manifest.groupby("sha256", sort=False)
    }
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    converted = mask_rows = box_rows = 0
    json_paths = sorted(json_root.glob("*.json"))
    overlay_count = 0

    # Establish annotation agreement before creating any per-sample artifacts. Exact-image
    # duplicates can carry different drawings, so processing them sequentially is unsafe.
    converted_by_hash: dict[str, list[tuple[Path, Path, tuple[Any, ...]]]] = {}
    for json_path in json_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                continue
            image_path = image_root / _safe_image_name(payload, json_path)
            if not image_path.is_file():
                continue
            source_hash = _sha256(image_path)
            result = convert_labelme(payload, image_path, allowed_labels=allowed_labels)
            converted_by_hash.setdefault(source_hash, []).append(
                (json_path, image_path, _spatial_signature(result))
            )
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, TypeError, ValueError):
            continue
    conflicting_paths: set[Path] = set()
    canonical_for: dict[Path, Path] = {}
    aliases_for: dict[Path, list[Path]] = {}
    for candidates in converted_by_hash.values():
        signatures = {item[2] for item in candidates}
        if len(signatures) > 1:
            conflicting_paths.update(item[0] for item in candidates)
            continue
        canonical = candidates[0][0]
        aliases = [item[0] for item in candidates[1:]]
        aliases_for[canonical] = aliases
        for alias in aliases:
            canonical_for[alias] = canonical

    for json_path in json_paths:
        image_path: Path | None = None
        if json_path in conflicting_paths:
            issues.append(
                _issue(
                    json_path,
                    None,
                    "conflicting_source_annotations",
                    "Exact source-image duplicates contain different normalized geometry",
                )
            )
            continue
        if json_path in canonical_for:
            converted += 1
            issues.append(
                _issue(
                    json_path,
                    None,
                    "equivalent_duplicate_annotation",
                    f"Equivalent to canonical annotation {canonical_for[json_path].name}",
                )
            )
            continue
        try:
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                issues.append(_issue(json_path, None, "invalid_json", str(exc)))
                continue
            if not isinstance(payload, dict):
                issues.append(
                    _issue(json_path, None, "invalid_json", "LabelMe payload must be an object")
                )
                continue
            try:
                image_name = _safe_image_name(payload, json_path)
            except ValueError as exc:
                issues.append(_issue(json_path, None, str(exc), str(exc)))
                continue
            image_path = image_root / image_name
            if not image_path.is_file():
                issues.append(
                    _issue(
                        json_path,
                        image_path,
                        "missing_source_image",
                        "Paired source image is missing",
                    )
                )
                continue
            source_hash = _sha256(image_path)
            matches = by_hash.get(source_hash, [])
            if not matches:
                issues.append(
                    _issue(
                        json_path,
                        image_path,
                        "no_eligible_exact_hash_match",
                        "No eligible Main row has the source image SHA-256",
                    )
                )
                continue
            try:
                result = convert_labelme(payload, image_path, allowed_labels=allowed_labels)
            except (TypeError, ValueError, OSError) as exc:
                issues.append(_issue(json_path, image_path, "invalid_labelme", str(exc)))
                continue
            verified_matches = []
            for match in matches:
                try:
                    main_path = context.image_path(match)
                    actual = _sha256(main_path)
                except OSError as exc:
                    issues.append(
                        _issue(
                            json_path,
                            image_path,
                            "eligible_main_unreadable",
                            f"{match['sample_id']}: {exc}",
                        )
                    )
                    continue
                except ValueError as exc:
                    issues.append(
                        _issue(
                            json_path,
                            image_path,
                            "eligible_main_hash_mismatch",
                            f"{match['sample_id']}: {exc}",
                        )
                    )
                    continue
                if actual != source_hash:
                    issues.append(
                        _issue(
                            json_path,
                            image_path,
                            "eligible_main_hash_mismatch",
                            f"{match['sample_id']}: Main bytes no longer match "
                            "its manifest SHA-256",
                        )
                    )
                    continue
                verified_matches.append(match)
            if not verified_matches:
                continue
            converted += 1
            source_json_hash = _sha256(json_path)
            for match in verified_matches:
                saved = save_spatial(result, generation_root, str(match["sample_id"]))
                mask_path = saved["mask_path"]
                if mask_path is not None:
                    mask_rows += 1
                if result["boxes"]:
                    box_rows += 1
                if overlay_count < 12:
                    render_overlay(
                        image_path,
                        result,
                        generation_root / "overlays" / f"{match['sample_id']}.png",
                    )
                    overlay_count += 1
                rows.append(
                    {
                        "sample_id": match["sample_id"],
                        "image_id": match["image_id"],
                        "source_sha256": source_hash,
                        "source_json_sha256": source_json_hash,
                        "mask_path": str(mask_path.resolve()) if mask_path else "",
                        "box_path": str(saved["box_path"].resolve()),
                        "width": result["width"],
                        "height": result["height"],
                        "source_json_path": str(json_path.resolve()),
                        "source_json_aliases": json.dumps(
                            [str(path.resolve()) for path in aliases_for.get(json_path, [])]
                        ),
                        "assigned_split": match.get("assigned_split", ""),
                        "subtype_label": match.get("subtype_label", ""),
                        "risk_label": match.get("risk_label", ""),
                        "status": "mechanically_validated",
                        "provenance": "source_dataset",
                        "reviewer": "",
                        "clinical_verification": "Unavailable",
                    }
                )
        except OSError as exc:
            issues.append(_issue(json_path, image_path, "source_io_error", str(exc)))

    frame = (
        pd.DataFrame(rows, columns=_MASTER_COLUMNS).sort_values("sample_id").reset_index(drop=True)
    )
    issue_frame = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    generation_csv = generation_root / "source_annotations.csv"
    generation_issues = generation_root / "source_annotation_issues.csv"
    master_csv = context.root / "source_annotations.csv"
    issues_csv = context.root / "source_annotation_issues.csv"
    _write_csv(frame, generation_csv)
    _write_csv(issue_frame, generation_issues)
    _write_csv(frame, master_csv)
    _write_csv(issue_frame, issues_csv)
    invalid_codes = {
        "invalid_json",
        "unsafe_image_basename",
        "json_image_basename_mismatch",
        "missing_source_image",
        "invalid_labelme",
        "source_io_error",
        "conflicting_source_annotations",
    }
    unmatched_codes = {
        "no_eligible_exact_hash_match",
        "eligible_main_unreadable",
        "eligible_main_hash_mismatch",
    }
    return {
        "source_json_count": len(json_paths),
        "converted_json_count": converted,
        "exported_sample_rows": len(frame),
        "invalid_json_count": len(
            {item["source_json_path"] for item in issues if item["error_code"] in invalid_codes}
        ),
        "unmatched_json_count": len(
            {item["source_json_path"] for item in issues if item["error_code"] in unmatched_codes}
        ),
        "mask_rows": mask_rows,
        "box_rows": box_rows,
        "generation_root": str(generation_root.resolve()),
        "generation_csv": str(generation_csv.resolve()),
        "master_csv": str(master_csv.resolve()),
        "issues_csv": str(issues_csv.resolve()),
        "generation_issues_csv": str(generation_issues.resolve()),
    }


def validate_source_import(config_path: Path) -> dict[str, Any]:
    """Reconvert the latest source index and verify hashes, geometry and derived files."""
    config_path = Path(config_path).resolve()
    context = load_annotation_context(config_path)
    data = _data_config(config_path, context.config)
    source_root = _source_root(context, data)
    source_json_root = (source_root / "Annotations").resolve()
    imports_root = (context.root / "source_imports").resolve()
    master_csv = (context.root / "source_annotations.csv").resolve()
    frame = pd.read_csv(master_csv, keep_default_na=False)
    missing = set(_MASTER_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Source annotation index missing columns: {sorted(missing)}")
    manifest = context.manifest.set_index("sample_id", drop=False)
    allowed = list(context.schema.get("subtype", {}).get("classes", []))
    allowed.extend(context.config.get("spatial_labels", ["lesion"]))
    allowed = list(dict.fromkeys(allowed))
    issues: list[dict[str, str]] = []

    def record(sample_id: str, code: str, error: str) -> None:
        issues.append({"sample_id": sample_id, "error_code": code, "error": error})

    duplicate_ids = sorted(frame.loc[frame.sample_id.duplicated(False), "sample_id"].unique())
    for sample_id in duplicate_ids:
        record(str(sample_id), "duplicate_sample_id", "Latest index contains duplicate rows")

    generation_roots = {
        Path(value).resolve().parent.parent for value in frame.box_path if str(value)
    }
    if not generation_roots and frame.empty:
        generation_indexes = list(imports_root.glob("*/source_annotations.csv"))
        if generation_indexes:
            latest = max(generation_indexes, key=lambda path: path.stat().st_mtime_ns)
            generation_roots = {latest.parent.resolve()}
    if len(generation_roots) != 1:
        record("", "generation_root_mismatch", "Index must reference one immutable generation")
    else:
        generation_csv = next(iter(generation_roots)) / "source_annotations.csv"
        try:
            immutable = pd.read_csv(generation_csv, keep_default_na=False)
            if list(immutable.columns) != list(frame.columns) or not immutable.equals(frame):
                record(
                    "",
                    "latest_generation_index_mismatch",
                    "Latest index differs from its immutable generation index",
                )
        except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
            record("", "immutable_generation_index_unreadable", str(exc))

    for row in frame.to_dict("records"):
        sample_id = str(row["sample_id"])
        try:
            if sample_id not in manifest.index:
                raise ValueError("sample_missing_from_eligible_manifest")
            manifest_row = manifest.loc[sample_id].to_dict()
            main = context.image_path(manifest_row)
            if _sha256(main) != row["source_sha256"]:
                raise ValueError("main_hash_mismatch")
            expected_metadata = {
                "image_id": manifest_row["image_id"],
                "assigned_split": manifest_row.get("assigned_split", ""),
                "subtype_label": manifest_row.get("subtype_label", ""),
                "risk_label": manifest_row.get("risk_label", ""),
                "status": "mechanically_validated",
                "provenance": "source_dataset",
                "reviewer": "",
                "clinical_verification": "Unavailable",
            }
            if any(str(row[key]) != str(value) for key, value in expected_metadata.items()):
                raise ValueError("source_metadata_mismatch")
            json_path = Path(row["source_json_path"]).resolve()
            if json_path.parent != source_json_root:
                raise ValueError("source_json_path_outside_btxrd")
            if _sha256(json_path) != row["source_json_sha256"]:
                raise ValueError("source_json_hash_mismatch")
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            image_name = _safe_image_name(payload, json_path)
            image_path = json_path.parent.parent / "images" / image_name
            if _sha256(image_path) != row["source_sha256"]:
                raise ValueError("source_image_hash_mismatch")
            result = convert_labelme(payload, image_path, allowed_labels=allowed)
            try:
                alias_values = json.loads(row["source_json_aliases"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError("invalid_source_json_aliases") from exc
            if not isinstance(alias_values, list) or any(
                not isinstance(value, str) for value in alias_values
            ):
                raise ValueError("invalid_source_json_aliases")
            for alias_value in alias_values:
                alias_path = Path(alias_value).resolve()
                if alias_path.parent != source_json_root:
                    raise ValueError("source_json_path_outside_btxrd")
                alias_payload = json.loads(alias_path.read_text(encoding="utf-8-sig"))
                alias_image_name = _safe_image_name(alias_payload, alias_path)
                alias_image = alias_path.parent.parent / "images" / alias_image_name
                if _sha256(alias_image) != row["source_sha256"]:
                    raise ValueError("source_image_hash_mismatch")
                alias_result = convert_labelme(alias_payload, alias_image, allowed_labels=allowed)
                if _spatial_signature(alias_result) != _spatial_signature(result):
                    raise ValueError("conflicting_source_annotations")
            if (result["width"], result["height"]) != (int(row["width"]), int(row["height"])):
                raise ValueError("geometry_dimension_mismatch")
            box_path = Path(row["box_path"]).resolve()
            generation_root = box_path.parent.parent
            if (
                box_path.parent.name != "boxes"
                or generation_root.parent != imports_root
                or box_path.name != f"{sample_id}.json"
            ):
                raise ValueError("artifact_path_outside_generation")
            box_payload = json.loads(box_path.read_text(encoding="utf-8"))
            if (
                box_payload.get("sample_id") != sample_id
                or box_payload.get("width") != result["width"]
                or box_payload.get("height") != result["height"]
                or box_payload.get("boxes") != result["boxes"]
                or box_payload.get("labels") != result["labels"]
            ):
                raise ValueError("box_content_mismatch")
            mask_path = str(row["mask_path"])
            if result["mask"] is None:
                if mask_path:
                    raise ValueError("unexpected_mask_path")
            else:
                if not mask_path:
                    raise ValueError("missing_mask_path")
                resolved_mask = Path(mask_path).resolve()
                if (
                    resolved_mask.parent != generation_root / "masks"
                    or resolved_mask.name != f"{sample_id}.png"
                ):
                    raise ValueError("artifact_path_outside_generation")
                with Image.open(resolved_mask) as opened:
                    actual_mask = np.asarray(opened, dtype=np.uint8)
                if actual_mask.shape != result["mask"].shape or not np.array_equal(
                    actual_mask, result["mask"]
                ):
                    raise ValueError("mask_content_mismatch")
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            code = str(exc) if str(exc) and " " not in str(exc) else "validation_error"
            record(sample_id, code, str(exc))
    return {
        "valid": not issues,
        "checked_rows": len(frame),
        "issue_count": len(issues),
        "issues": issues,
        "master_csv": str(master_csv),
    }
