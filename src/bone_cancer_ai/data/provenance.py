"""Read local source metadata solely to verify Main labels and available supervision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bone_cancer_ai.data.schema import canonical_source_id

PROVENANCE_COLUMNS = (
    "canonical_source_id",
    "source_name",
    "source_record",
    "source_sha256",
    "binary_semantic",
    "risk_label",
    "subtype_label",
    "has_box",
    "has_mask",
    "source_issue",
)

SUBTYPE_COLUMNS = (
    "giant cell tumor",
    "multiple osteochondromas",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "other bt",
    "other mt",
    "simple bone cyst",
    "synovial osteochondroma",
)
SOURCE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _path(config: dict[str, Any], project_root: Path, key: str) -> Path | None:
    value = config.get("provenance", {}).get(key)
    if not value:
        return None
    result = Path(value)
    return (result if result.is_absolute() else project_root / result).resolve()


def _folder_rows(root: Path, source_name: str, classes: dict[str, tuple[str, str]]) -> list[dict]:
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*")):
        class_name = path.parent.name.lower()
        if (
            not path.is_file()
            or path.suffix.lower() not in SOURCE_IMAGE_EXTENSIONS
            or class_name not in classes
        ):
            continue
        semantic, risk = classes[class_name]
        rows.append(
            {
                "canonical_source_id": canonical_source_id(path.name),
                "source_name": source_name,
                "source_record": path.relative_to(root).as_posix(),
                "source_sha256": _sha256(path),
                "binary_semantic": semantic,
                "risk_label": risk,
                "subtype_label": "",
                "has_box": False,
                "has_mask": False,
                "source_issue": "",
            }
        )
    return rows


def _annotation_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    annotation_root = root / "Annotations"
    if not annotation_root.is_dir():
        return inventory
    for path in sorted(annotation_root.glob("*.json")):
        entry = {"has_box": False, "has_mask": False, "source_issue": ""}
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            shapes = payload.get("shapes", [])
            entry["has_box"] = any(shape.get("shape_type") == "rectangle" for shape in shapes)
            entry["has_mask"] = any(shape.get("shape_type") == "polygon" for shape in shapes)
        except Exception as exc:
            entry["source_issue"] = f"annotation {type(exc).__name__}: {exc}"
        inventory[canonical_source_id(path.name)] = entry
    return inventory


def _one_hot_label(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, str]:
    active = [
        name
        for name in columns
        if name in row.index and pd.to_numeric(row[name], errors="coerce") == 1
    ]
    if len(active) == 1:
        return active[0], ""
    return "", f"subtype_one_hot_count={len(active)}"


def _btxrd_rows(root: Path) -> list[dict]:
    workbook = root / "dataset.xlsx"
    if not workbook.is_file():
        return []
    frame = pd.read_excel(workbook, sheet_name="Sheet1")
    required = {"image_id", "tumor", "benign", "malignant"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BTXRD workbook missing columns: {missing}")
    annotations = _annotation_inventory(root)
    source_images: dict[str, list[Path]] = {}
    image_root = root / "images"
    if image_root.is_dir():
        for path in sorted(image_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SOURCE_IMAGE_EXTENSIONS:
                source_images.setdefault(canonical_source_id(path.name), []).append(path)
    rows = []
    for _, row in frame.iterrows():
        image_id = str(row["image_id"]).strip()
        canonical = canonical_source_id(image_id)
        tumor = pd.to_numeric(row["tumor"], errors="coerce")
        semantic = "bone_tumour" if tumor == 1 else "normal" if tumor == 0 else ""
        benign = pd.to_numeric(row["benign"], errors="coerce") == 1
        malignant = pd.to_numeric(row["malignant"], errors="coerce") == 1
        risk = (
            "benign"
            if benign and not malignant
            else "malignant"
            if malignant and not benign
            else ""
        )
        subtype, subtype_issue = _one_hot_label(row, SUBTYPE_COLUMNS) if tumor == 1 else ("", "")
        spatial = annotations.get(canonical, {})
        issues = [value for value in (subtype_issue, spatial.get("source_issue", "")) if value]
        image_matches = source_images.get(canonical, [])
        source_sha256 = _sha256(image_matches[0]) if len(image_matches) == 1 else ""
        if len(image_matches) != 1:
            issues.append(f"btxrd_image_count={len(image_matches)}")
        if tumor == 1 and benign == malignant:
            issues.append(f"risk_one_hot_count={int(benign) + int(malignant)}")
        rows.append(
            {
                "canonical_source_id": canonical,
                "source_name": "btxrd_workbook",
                "source_record": image_id,
                "source_sha256": source_sha256,
                "binary_semantic": semantic,
                "risk_label": risk,
                "subtype_label": subtype,
                "has_box": bool(spatial.get("has_box", False)),
                "has_mask": bool(spatial.get("has_mask", False)),
                "source_issue": "; ".join(issues),
            }
        )
    return rows


def _detection_rows(root: Path) -> list[dict]:
    yaml_path = root / "data.yaml"
    label_root = root / "labels"
    if not yaml_path.is_file() or not label_root.is_dir():
        return []
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    names = metadata.get("names", []) if isinstance(metadata, dict) else []
    if not isinstance(names, list):
        raise ValueError("Detection data.yaml names must be a list")
    rows = []
    for path in sorted(label_root.rglob("*.txt")):
        classes: set[int] = set()
        issues = []
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = int(line.split()[0])
                if value < 0 or value >= len(names):
                    raise ValueError(f"class {value} outside names")
                classes.add(value)
            except Exception as exc:
                issues.append(f"line {number}: {exc}")
        subtype = names[next(iter(classes))] if len(classes) == 1 else ""
        if len(classes) != 1:
            issues.append(f"unique_class_count={len(classes)}")
        relative = path.relative_to(label_root)
        image_dir = root / "images" / relative.parent
        image_matches = (
            sorted(
                candidate
                for candidate in image_dir.glob(f"{path.stem}.*")
                if candidate.is_file() and candidate.suffix.lower() in SOURCE_IMAGE_EXTENSIONS
            )
            if image_dir.is_dir()
            else []
        )
        source_sha256 = _sha256(image_matches[0]) if len(image_matches) == 1 else ""
        if len(image_matches) != 1:
            issues.append(f"detection_image_count={len(image_matches)}")
        rows.append(
            {
                "canonical_source_id": canonical_source_id(path.name),
                "source_name": "subtype_detection",
                "source_record": path.relative_to(root).as_posix(),
                "source_sha256": source_sha256,
                "binary_semantic": "bone_tumour",
                "risk_label": "",
                "subtype_label": subtype,
                "has_box": bool(classes),
                "has_mask": False,
                "source_issue": "; ".join(issues),
            }
        )
    return rows


def load_local_provenance(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load configured local evidence without reading labels from Main filenames."""
    project_root = Path(project_root).resolve()
    rows: list[dict] = []
    binary = _path(config, project_root, "roboflow_binary_root")
    if binary is not None:
        rows.extend(
            _folder_rows(
                binary,
                "roboflow_binary_folders",
                {"normal": ("normal", ""), "cancer": ("bone_tumour", "")},
            )
        )
    three = _path(config, project_root, "three_class_root")
    if three is not None:
        rows.extend(
            _folder_rows(
                three,
                "three_class_folders",
                {
                    "normal": ("normal", ""),
                    "benign": ("bone_tumour", "benign"),
                    "malignant": ("bone_tumour", "malignant"),
                },
            )
        )
    btxrd = _path(config, project_root, "btxrd_root")
    if btxrd is not None:
        rows.extend(_btxrd_rows(btxrd))
    detection = _path(config, project_root, "subtype_detection_root")
    if detection is not None:
        rows.extend(_detection_rows(detection))
    result = pd.DataFrame(rows, columns=PROVENANCE_COLUMNS)
    if not result.empty:
        result = result.sort_values(
            ["canonical_source_id", "source_name", "source_record"]
        ).reset_index(drop=True)
    return result
