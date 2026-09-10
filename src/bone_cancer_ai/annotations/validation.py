"""Revalidate persisted human annotations against their immutable source identities."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from bone_cancer_ai.annotations.schema import validate_metadata
from bone_cancer_ai.annotations.spatial import convert_labelme


def _validate_audit(record: dict) -> None:
    if (
        record.get("provenance") != "project_annotation"
        or record.get("clinical_verification") != "Unavailable"
    ):
        raise ValueError("Invalid project annotation provenance")
    try:
        timestamp = datetime.fromisoformat(record["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid annotation timestamp") from exc
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("Annotation timestamp must have explicit UTC timezone")


def validate_project_annotations(ctx) -> dict:
    """Check every metadata/spatial revision, including retained historical artifacts."""
    identities = ctx.manifest.set_index("sample_id").to_dict("index")
    metadata_count = spatial_count = 0
    for sample_id in ctx.store.export_frame().sample_id:
        if sample_id not in identities:
            raise ValueError(f"Reviewed sample no longer eligible: {sample_id}")
        row = identities[sample_id]
        ctx.image_path(row)
        for expected, record in enumerate(ctx.store.history(sample_id), start=1):
            _validate_audit(record)
            if (
                record["source_sha256"] != row["sha256"]
                or record["image_id"] != row["image_id"]
                or record["revision"] != expected
            ):
                raise ValueError("Stored metadata identity/revision mismatch")
            fields = (
                "binary_review",
                "subtype_label",
                "risk_label",
                "concepts",
                "reviewer",
                "review_status",
                "notes",
            )
            validate_metadata(
                {key: record[key] for key in fields}, ctx.schema, ctx.config["concept_names"]
            )
            metadata_count += 1
    for revision in (ctx.root / "spatial_reviews").glob("*/revision.json"):
        record = json.loads(revision.read_text(encoding="utf-8"))
        _validate_audit(record)
        if record.get("revision_id") != revision.parent.name:
            raise ValueError("Spatial revision identity does not match directory")
        if record["sample_id"] not in identities:
            raise ValueError("Spatial sample no longer eligible")
        row = identities[record["sample_id"]]
        source = ctx.image_path(row)
        if (
            record["source_sha256"] != row["sha256"]
            or record["image_id"] != row["image_id"]
            or not record["reviewer"].strip()
        ):
            raise ValueError("Spatial identity or reviewer mismatch")

        def artifact(key, record=record, revision=revision):
            path = Path(record[key]).resolve(strict=True)
            if not path.is_relative_to(revision.parent.resolve()):
                raise ValueError("Spatial artifact escapes revision directory")
            return path

        data = artifact("labelme_json_path").read_bytes()
        if hashlib.sha256(data).hexdigest() != record["labelme_json_sha256"]:
            raise ValueError("Spatial JSON hash changed")
        payload = json.loads(data)
        # Identity was verified above; the preserved JSON names the opaque staged copy.
        payload["imagePath"] = source.name
        expected = convert_labelme(payload, source, ctx.config["spatial_labels"])
        boxes = json.loads(artifact("box_path").read_text(encoding="utf-8"))
        if (
            boxes["boxes"] != expected["boxes"]
            or boxes["width"] != expected["width"]
            or boxes["height"] != expected["height"]
            or boxes["labels"] != expected["labels"]
        ):
            raise ValueError("Stored boxes differ from source JSON")
        if expected["mask"] is not None:
            with Image.open(artifact("mask_path")) as mask:
                if not np.array_equal(np.asarray(mask), expected["mask"]):
                    raise ValueError("Stored mask differs from source JSON")
        elif record["mask_path"]:
            raise ValueError("Boxes-only annotation contains a fabricated mask")
        overlay = artifact("overlay_path")
        if hashlib.sha256(overlay.read_bytes()).hexdigest() != record["overlay_sha256"]:
            raise ValueError("Spatial overlay hash changed")
        with Image.open(overlay) as opened:
            opened.load()
            if opened.size != (expected["width"], expected["height"]):
                raise ValueError("Spatial overlay dimensions changed")
        spatial_count += 1
    return {
        "metadata_revisions": metadata_count,
        "spatial_revisions": spatial_count,
        "clinical_verification": "Unavailable",
        "status": "passed",
    }
