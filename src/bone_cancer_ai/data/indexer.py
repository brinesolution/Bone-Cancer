"""Read-only image inventory with labels, hashes, decode metadata and quality signals."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image, ImageOps

from bone_cancer_ai.data.quality import analyze_pixels
from bone_cancer_ai.data.schema import (
    ISSUE_COLUMNS,
    MANIFEST_COLUMNS,
    QualityStatus,
    canonical_source_id,
    load_main_labels,
    record_id,
)


def inspect_image(
    path: Path,
    official_split: str,
    binary_label: int | None,
    thresholds: dict[str, Any],
    *,
    image_id: str | None = None,
) -> dict[str, Any]:
    """Inspect one image without modifying it; decode failures become manifest rows."""
    path = Path(path).resolve()
    image_id = image_id or path.name
    base = {
        "record_id": record_id(official_split, image_id),
        "image_id": image_id,
        "absolute_path": str(path),
        "official_split": official_split,
        "extension": path.suffix.lower(),
        "width": None,
        "height": None,
        "channels": None,
        "pil_mode": "",
        "file_size": path.stat().st_size,
        "sha256": "",
        "phash": "",
        "canonical_source_id": canonical_source_id(image_id),
        "binary_label": binary_label,
        "binary_semantic": "",
        "subtype_label": "",
        "risk_label": "",
        "has_box": False,
        "has_mask": False,
        "has_concepts": False,
        "is_rgb_encoded_grayscale": False,
        "intensity_mean": None,
        "intensity_std": None,
        "entropy": None,
        "dark_fraction": None,
        "bright_fraction": None,
        "quality_status": str(QualityStatus.UNREADABLE),
        "exclude_reason": "",
        "decode_error": "",
    }
    try:
        with path.open("rb") as stream:
            base["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with Image.open(path) as opened:
            base["width"], base["height"] = opened.size
            base["pil_mode"] = opened.mode
            base["channels"] = len(opened.getbands())
            image = ImageOps.exif_transpose(opened)
            image.load()
        base["phash"] = str(imagehash.phash(image))
        base.update(analyze_pixels(image, thresholds))
    except Exception as exc:
        base["quality_status"] = str(QualityStatus.UNREADABLE)
        base["decode_error"] = f"{type(exc).__name__}: {exc}"
    return base


def index_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Index configured splits and report bidirectional image/label mismatches."""
    dataset_root = Path(config["dataset_root"]).resolve()
    splits = list(config["official_splits"])
    extensions = {str(value).lower() for value in config["image_extensions"]}
    inventory = load_main_labels(dataset_root, splits)
    issues = inventory.issues.to_dict("records")
    label_lookup = (
        inventory.labels.drop_duplicates("record_id", keep="first")
        .set_index("record_id")["binary_label"]
        .to_dict()
        if not inventory.labels.empty
        else {}
    )
    items: list[tuple[Path, str, str]] = []
    for split in splits:
        split_root = dataset_root / split
        for path in sorted(split_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                image_id = path.relative_to(split_root).as_posix()
                items.append((path, split, image_id))
    image_records = {record_id(split, image_id) for _, split, image_id in items}
    label_records = set(inventory.labels.loc[inventory.labels["record_id"].ne(""), "record_id"])
    for missing in sorted(image_records - label_records):
        split, image_id = missing.split("/", 1)
        issues.append(
            {
                "record_id": missing,
                "image_id": image_id,
                "official_split": split,
                "issue_type": "image_missing_label",
                "details": "Image has no CSV row",
                "source": "main_index",
            }
        )
    for missing in sorted(label_records - image_records):
        split, image_id = missing.split("/", 1)
        issues.append(
            {
                "record_id": missing,
                "image_id": image_id,
                "official_split": split,
                "issue_type": "label_missing_image",
                "details": "CSV row has no image file",
                "source": "main_index",
            }
        )
    quality = config.get("audit", {}).get("quality", {})

    def inspect(item: tuple[Path, str, str]) -> dict[str, Any]:
        path, split, image_id = item
        label = label_lookup.get(record_id(split, image_id))
        value = None if label is None or pd.isna(label) else int(label)
        return inspect_image(path, split, value, quality, image_id=image_id)

    workers = max(1, int(config.get("audit", {}).get("workers", 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(inspect, items))
    master = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not master.empty:
        order = {split: index for index, split in enumerate(splits)}
        master["_split_order"] = master["official_split"].map(order)
        master = master.sort_values(["_split_order", "image_id"]).drop(columns="_split_order")
        master = master.reset_index(drop=True)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    summary = {
        "images_indexed": len(master),
        "label_rows": len(inventory.labels),
        "issues": len(issue_frame),
        "unreadable_images": int(master["quality_status"].eq("unreadable").sum()),
        "source_csvs": inventory.csv_files,
    }
    return master, issue_frame, summary
