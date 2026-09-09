"""Conservative, non-destructive screening of decoded image pixels."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from bone_cancer_ai.data.schema import QualityStatus, opaque_id


def analyze_pixels(image: Image.Image, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Measure a small grayscale preview and flag uniform scans for manual review."""
    size = int(thresholds.get("thumbnail_size", 256))
    preview = image.copy()
    preview.thumbnail((size, size), Image.Resampling.BILINEAR)
    rgb_grayscale = False
    if image.mode in {"RGB", "RGBA"}:
        rgb = np.asarray(preview.convert("RGB"), dtype=np.int16)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        rgb_grayscale = bool(spread.max(initial=0) <= thresholds.get("rgb_grayscale_tolerance", 1))
    values = np.asarray(preview.convert("L"), dtype=np.uint8)
    mean = float(values.mean())
    std = float(values.std())
    counts = np.bincount(values.ravel(), minlength=256)
    probabilities = counts[counts > 0] / values.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    dark = float(np.mean(values <= thresholds.get("dark_pixel_threshold", 5)))
    bright = float(np.mean(values >= thresholds.get("bright_pixel_threshold", 250)))
    if std <= thresholds.get("blank_std_threshold", 0.5) or entropy <= thresholds.get(
        "blank_entropy_threshold", 0.1
    ):
        status = QualityStatus.BLANK
    elif std <= thresholds.get("near_blank_std_threshold", 3.0) or entropy <= thresholds.get(
        "near_blank_entropy_threshold", 1.0
    ):
        status = QualityStatus.NEAR_BLANK
    else:
        status = QualityStatus.READABLE
    return {
        "is_rgb_encoded_grayscale": rgb_grayscale,
        "intensity_mean": mean,
        "intensity_std": std,
        "entropy": entropy,
        "dark_fraction": dark,
        "bright_fraction": bright,
        "quality_status": str(status),
    }


def build_quality_report(master: pd.DataFrame) -> pd.DataFrame:
    """Select per-image quality evidence without making exclusion decisions."""
    columns = [
        "record_id",
        "image_id",
        "official_split",
        "width",
        "height",
        "channels",
        "pil_mode",
        "is_rgb_encoded_grayscale",
        "intensity_mean",
        "intensity_std",
        "entropy",
        "dark_fraction",
        "bright_fraction",
        "quality_status",
        "decode_error",
    ]
    return master.reindex(columns=columns).copy()


def _duplicate_membership(duplicates: pd.DataFrame) -> pd.Series:
    required = {
        "duplicate_group",
        "record_id",
        "other_record_id",
        "match_type",
        "label_conflict",
    }
    missing = required - set(duplicates.columns)
    if missing:
        raise ValueError(f"Duplicate report missing columns: {sorted(missing)}")
    members = pd.concat(
        [
            duplicates[["duplicate_group", "record_id"]].rename(columns={"record_id": "record"}),
            duplicates[["duplicate_group", "other_record_id"]].rename(
                columns={"other_record_id": "record"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    conflicts = members.groupby("record")["duplicate_group"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A record appears in multiple duplicate groups")
    return members.drop_duplicates("record").set_index("record")["duplicate_group"]


def _append_reason(current: str, reason: str) -> str:
    values = [value for value in str(current).split(";") if value]
    if reason and reason not in values:
        values.append(reason)
    return ";".join(values)


def build_clean_manifest(
    master: pd.DataFrame,
    duplicates: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply deterministic exclusion and held-out-priority rules to audited rows."""
    required = {
        "record_id",
        "official_split",
        "binary_label",
        "quality_status",
        "exclude_reason",
    }
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master manifest missing columns: {sorted(missing)}")
    if master["record_id"].isna().any() or master["record_id"].duplicated().any():
        raise ValueError("Master record_id values must be present and unique")
    heldout_priority = list(policy.get("heldout_priority", ["test", "val", "train"]))
    if set(heldout_priority) != {"test", "val", "train"}:
        raise ValueError("heldout_priority must contain test, val, and train exactly once")
    excluded_quality = set(policy.get("excluded_quality_statuses", ["unreadable", "blank"]))
    exact_action = str(policy.get("exact_label_conflict_action", "exclude_group"))
    if exact_action != "exclude_group":
        raise ValueError("exact_label_conflict_action must be exclude_group")
    perceptual_action = str(
        policy.get("perceptual_label_conflict_action", "retain_group_and_flag")
    )
    if perceptual_action != "retain_group_and_flag":
        raise ValueError(
            "perceptual_label_conflict_action must be retain_group_and_flag"
        )

    clean = master.copy()
    membership = _duplicate_membership(duplicates)
    clean["duplicate_group"] = clean["record_id"].map(membership).fillna("").astype("string")
    clean["sample_id"] = clean["record_id"].map(lambda value: opaque_id("sample", value))
    clean["split_group_id"] = [
        str(group) if group else opaque_id("single", record)
        for record, group in zip(clean["record_id"], clean["duplicate_group"], strict=True)
    ]
    if clean["split_group_id"].duplicated().any():
        duplicate_ids = clean.loc[clean["split_group_id"].duplicated(False), "split_group_id"]
        duplicate_groups = clean.loc[clean["split_group_id"].isin(duplicate_ids), "duplicate_group"]
        if duplicate_groups.eq("").any():
            raise ValueError("Opaque singleton split_group_id collision")

    group_splits = clean.groupby("split_group_id")["official_split"].agg(set)
    target_official = {
        group: next(split for split in heldout_priority if split in splits)
        for group, splits in group_splits.items()
    }
    exact_conflict_groups = set(
        duplicates.loc[
            duplicates["label_conflict"].fillna(False).astype(bool)
            & duplicates["match_type"].astype(str).str.contains(
                "exact_sha256", regex=False
            ),
            "duplicate_group",
        ].astype(str)
    )
    perceptual_conflict_groups = set(
        duplicates.loc[
            duplicates["label_conflict"].fillna(False).astype(bool)
            & duplicates["match_type"].astype(str).str.contains(
                "perceptual_phash", regex=False
            )
            & ~duplicates["duplicate_group"].astype(str).isin(exact_conflict_groups),
            "duplicate_group",
        ].astype(str)
    )

    clean["clean_pool"] = ""
    clean["assigned_split"] = ""
    clean["review_required"] = False
    clean["review_reason"] = ""
    clean["split_correction_reason"] = ""
    source_exclusions = clean["exclude_reason"].fillna("").astype(str)
    clean["exclude_reason"] = source_exclusions

    for index, row in clean.iterrows():
        reason = str(row["exclude_reason"])
        review_reason = ""
        group = str(row["split_group_id"])
        duplicate_group = str(row["duplicate_group"])
        status = str(row["quality_status"])
        if not reason and status in excluded_quality:
            reason = f"quality_{status}"
        if not reason and pd.isna(row["binary_label"]):
            reason = "missing_binary_label"
        if not reason and duplicate_group in exact_conflict_groups:
            reason = "exact_duplicate_label_conflict"
        target = target_official[group]
        if not reason and row["official_split"] != target:
            reason = "cross_split_duplicate_lower_priority"
            clean.at[index, "split_correction_reason"] = (
                f"group_retained_in_official_{target}"
            )
        if status == "near_blank":
            review_reason = _append_reason(review_reason, "quality_near_blank")
        if duplicate_group in perceptual_conflict_groups:
            review_reason = _append_reason(review_reason, "perceptual_label_conflict")
        if reason:
            clean.at[index, "clean_pool"] = "excluded"
            clean.at[index, "assigned_split"] = "excluded"
            clean.at[index, "exclude_reason"] = reason
        else:
            pool = "train_pool" if target == "train" else target
            clean.at[index, "clean_pool"] = pool
            clean.at[index, "assigned_split"] = pool
        clean.at[index, "review_reason"] = review_reason
        clean.at[index, "review_required"] = bool(review_reason)

    clean["is_eligible"] = clean["clean_pool"].ne("excluded")
    leading = [
        "sample_id",
        "record_id",
        "image_id",
        "absolute_path",
        "official_split",
        "assigned_split",
        "clean_pool",
        "duplicate_group",
        "split_group_id",
        "binary_label",
        "is_eligible",
        "exclude_reason",
        "review_required",
        "review_reason",
        "split_correction_reason",
    ]
    clean = clean[leading + [column for column in clean.columns if column not in leading]]
    clean = clean.sort_values("record_id").reset_index(drop=True)
    exclusion_columns = [
        "sample_id",
        "record_id",
        "official_split",
        "duplicate_group",
        "split_group_id",
        "binary_label",
        "quality_status",
        "exclude_reason",
        "split_correction_reason",
        "review_required",
        "review_reason",
    ]
    exclusions = clean.loc[~clean["is_eligible"], exclusion_columns].reset_index(drop=True)

    transitions = Counter(
        f"{row.official_split}->{row.clean_pool}" for row in clean.itertuples(index=False)
    )
    summary = {
        "records": len(clean),
        "eligible_records": int(clean["is_eligible"].sum()),
        "excluded_records": int((~clean["is_eligible"]).sum()),
        "clean_pool_counts": {
            str(key): int(value) for key, value in clean["clean_pool"].value_counts().items()
        },
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in exclusions["exclude_reason"].value_counts().items()
        },
        "review_reason_counts": {
            str(key): int(value)
            for key, value in clean.loc[clean["review_required"], "review_reason"]
            .value_counts()
            .items()
        },
        "official_to_clean_counts": dict(sorted(transitions.items())),
        "official_records_reassigned_to_training": int(
            (
                clean["official_split"].isin(["val", "test"])
                & clean["clean_pool"].eq("train_pool")
            ).sum()
        ),
        "exact_conflict_groups_excluded": len(exact_conflict_groups),
        "perceptual_conflict_groups_flagged": len(perceptual_conflict_groups),
        "policy": {
            "heldout_priority": heldout_priority,
            "excluded_quality_statuses": sorted(excluded_quality),
            "cross_split_action": "exclude lower-priority official copies",
            "exact_label_conflict_action": exact_action,
            "perceptual_label_conflict_action": perceptual_action,
        },
    }
    return clean, exclusions, summary


def validate_clean_manifest(
    clean: pd.DataFrame, expected_records: set[str] | None = None
) -> dict[str, Any]:
    """Check all-record coverage and that eligible groups occupy only one clean pool."""
    required = {"record_id", "split_group_id", "clean_pool", "is_eligible"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Clean manifest missing columns: {sorted(missing)}")
    record_ids = set(clean["record_id"].astype(str))
    eligible = clean[clean["is_eligible"].astype(bool)]
    group_pool_overlap = int(
        (eligible.groupby("split_group_id")["clean_pool"].nunique() > 1).sum()
    )
    return {
        "records": len(clean),
        "eligible_records": len(eligible),
        "excluded_records": len(clean) - len(eligible),
        "record_ids_unique": bool(clean["record_id"].is_unique),
        "complete_coverage": expected_records is None or record_ids == expected_records,
        "group_pool_overlap": group_pool_overlap,
    }
