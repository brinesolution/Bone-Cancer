"""Local LabelMe orchestration and immutable reviewer-owned spatial revisions."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from bone_cancer_ai.annotations.labelme_bridge import import_staged, stage_labelme
from bone_cancer_ai.annotations.spatial import render_overlay, save_spatial

SPATIAL_COLUMNS = [
    "sample_id",
    "image_id",
    "source_sha256",
    "revision_id",
    "timestamp",
    "reviewer",
    "notes",
    "provenance",
    "clinical_verification",
    "mask_path",
    "box_path",
    "overlay_path",
    "overlay_sha256",
    "labelme_json_path",
    "labelme_json_sha256",
]


def prepare_labelme(ctx, row: dict) -> dict:
    """Stage once, then reopen the existing bound working copy for further edits."""
    image = ctx.image_path(row)
    binding = ctx.root / "labelme" / row["sample_id"] / f"{row['sample_id']}.binding.json"
    if binding.exists():
        data = json.loads(binding.read_text(encoding="utf-8"))
        if (
            data["source_sha256"] != row["sha256"]
            or Path(data["source_image_path"]).resolve() != image
        ):
            raise ValueError("Staged binding does not match this source record")
        return {
            "binding_path": binding,
            "image_path": binding.parent / data["staged_image"],
            "json_path": binding.parent / data["labelme_json"],
        }
    return stage_labelme(image, row["sample_id"], ctx.root, row["sha256"])


def launch_labelme(ctx, row: dict) -> subprocess.Popen:
    """Open the requested interactive native editor on a staged image only."""
    staged = prepare_labelme(ctx, row)
    labels = ctx.root / "labelme" / "labels.txt"
    labels.write_text("\n".join(ctx.config["spatial_labels"]) + "\n", encoding="utf-8")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "labelme",
            "--config",
            "{keep_prev: false, auto_save: true, with_image_data: false}",
            str(staged["image_path"]),
            "--output",
            str(staged["json_path"].parent),
            "--labels",
            str(labels),
            "--validate-label",
            "exact",
        ]
    )


def import_spatial_review(ctx, row: dict, reviewer: str, notes: str) -> dict:
    """Preserve an edited LabelMe JSON and its derived artifacts as a new revision."""
    if not reviewer.strip():
        raise ValueError("A reviewer name is required for spatial import")
    image = ctx.image_path(row)
    binding = ctx.root / "labelme" / row["sample_id"] / f"{row['sample_id']}.binding.json"
    result = import_staged(binding, ctx.config["spatial_labels"])
    if (
        result["source_sha256"] != row["sha256"]
        or result["sample_id"] != row["sample_id"]
        or result["source_image_path"] != image
    ):
        raise ValueError("Spatial binding does not match the selected source record")
    revision_id = uuid4().hex
    root = ctx.root / "spatial_reviews" / revision_id
    outputs = save_spatial(result, root, row["sample_id"])
    overlay = render_overlay(image, result, root / "overlay.png")
    json_bytes = result["labelme_json_bytes"]
    snapshot = root / "labelme.json"
    snapshot.write_bytes(json_bytes)
    record = {
        "sample_id": row["sample_id"],
        "image_id": row["image_id"],
        "source_sha256": row["sha256"],
        "revision_id": revision_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "reviewer": reviewer.strip(),
        "notes": notes,
        "provenance": "project_annotation",
        "clinical_verification": "Unavailable",
        "mask_path": str(outputs["mask_path"] or ""),
        "box_path": str(outputs["box_path"]),
        "overlay_path": str(overlay),
        "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        "labelme_json_path": str(snapshot),
        "labelme_json_sha256": hashlib.sha256(json_bytes).hexdigest(),
    }
    temporary = root / "revision.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / "revision.json")
    return record


def spatial_frame(root: Path) -> pd.DataFrame:
    """Return latest spatial revisions; all older revision files remain available."""
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (root / "spatial_reviews").glob("*/revision.json")
    ]
    frame = pd.DataFrame(records, columns=SPATIAL_COLUMNS)
    return frame.sort_values(["timestamp", "revision_id"]).drop_duplicates("sample_id", keep="last")


def export_annotations(ctx) -> pd.DataFrame:
    """Join latest human metadata and spatial reviews without copying source diagnoses."""
    metadata = ctx.store.export_frame()
    spatial = spatial_frame(ctx.root)
    return metadata.merge(
        spatial,
        how="outer",
        on=["sample_id", "image_id", "source_sha256"],
        suffixes=("", "_spatial"),
    )
