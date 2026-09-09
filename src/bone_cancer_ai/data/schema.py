"""Stable audit-table contracts and strict source-label readers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import pandas as pd

MANIFEST_COLUMNS = (
    "record_id",
    "image_id",
    "absolute_path",
    "official_split",
    "extension",
    "width",
    "height",
    "channels",
    "pil_mode",
    "file_size",
    "sha256",
    "phash",
    "canonical_source_id",
    "binary_label",
    "binary_semantic",
    "subtype_label",
    "risk_label",
    "has_box",
    "has_mask",
    "has_concepts",
    "is_rgb_encoded_grayscale",
    "intensity_mean",
    "intensity_std",
    "entropy",
    "dark_fraction",
    "bright_fraction",
    "quality_status",
    "exclude_reason",
    "decode_error",
)

ISSUE_COLUMNS = ("record_id", "image_id", "official_split", "issue_type", "details", "source")


class QualityStatus(StrEnum):
    """Non-destructive image-quality screening outcomes."""

    READABLE = "readable"
    NEAR_BLANK = "near_blank"
    BLANK = "blank"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class LabelInventory:
    """All parsed label rows and every structural issue found while reading them."""

    labels: pd.DataFrame
    issues: pd.DataFrame
    csv_files: dict[str, str]


def record_id(split: str, image_id: str) -> str:
    """Build the manifest's stable identifier without altering the source filename."""
    return f"{split}/{image_id}"


def canonical_source_id(name: str) -> str:
    """Remove only known terminal export suffixes used by the local Roboflow data."""
    stem = Path(str(name)).stem.lower()
    stem = re.sub(r"\.rf\.[0-9a-f]{32}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return stem


def _issue(split: str, issue_type: str, details: str, image_id: str = "") -> dict:
    return {
        "record_id": record_id(split, image_id) if image_id else "",
        "image_id": image_id,
        "official_split": split,
        "issue_type": issue_type,
        "details": details,
        "source": "main_label_csv",
    }


def load_main_labels(dataset_root: Path, splits: Sequence[str]) -> LabelInventory:
    """Read source CSVs without mutating them and report every invalid row/structure."""
    dataset_root = Path(dataset_root)
    frames: list[pd.DataFrame] = []
    issues: list[dict] = []
    csv_files: dict[str, str] = {}
    expected = ["image_id", "binary_label"]
    for split in splits:
        split_root = dataset_root / split
        csvs = sorted(split_root.glob("*.csv"))
        if len(csvs) != 1:
            issue_type = "missing_label_csv" if not csvs else "multiple_label_csv"
            issues.append(_issue(split, issue_type, f"Found {len(csvs)} CSV files"))
            continue
        csv_path = csvs[0]
        csv_files[split] = str(csv_path.resolve())
        try:
            frame = pd.read_csv(csv_path, dtype="string", keep_default_na=True)
        except Exception as exc:
            issues.append(_issue(split, "unreadable_label_csv", f"{type(exc).__name__}: {exc}"))
            continue
        columns = list(frame.columns)
        if columns != expected:
            issues.append(_issue(split, "unexpected_label_columns", repr(columns)))
        if not set(expected).issubset(columns):
            continue
        frame = frame[expected].copy()
        frame["image_id"] = frame["image_id"].str.strip()
        missing_id = frame["image_id"].isna() | frame["image_id"].eq("")
        for _ in frame.index[missing_id]:
            issues.append(_issue(split, "missing_image_id", "Blank image_id"))
        duplicate = frame["image_id"].notna() & frame["image_id"].duplicated(keep=False)
        for image_id in sorted(frame.loc[duplicate, "image_id"].dropna().unique()):
            count = int(frame["image_id"].eq(image_id).sum())
            issues.append(_issue(split, "duplicate_label_id", f"{count} rows", str(image_id)))
        numeric = pd.to_numeric(frame["binary_label"], errors="coerce")
        valid_label = numeric.isin([0, 1]) & numeric.notna()
        for idx in frame.index[~valid_label]:
            image_id = "" if pd.isna(frame.at[idx, "image_id"]) else str(frame.at[idx, "image_id"])
            value = frame.at[idx, "binary_label"]
            issues.append(_issue(split, "invalid_binary_label", repr(value), image_id))
        frame["binary_label"] = numeric.where(valid_label).astype("Int64")
        frame["official_split"] = split
        frame["record_id"] = [
            record_id(split, str(value)) if pd.notna(value) and str(value) else ""
            for value in frame["image_id"]
        ]
        frame["label_csv"] = str(csv_path.resolve())
        frames.append(
            frame[["record_id", "image_id", "binary_label", "official_split", "label_csv"]]
        )
    labels = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["record_id", "image_id", "binary_label", "official_split", "label_csv"]
        )
    )
    if not labels.empty:
        labels = labels.sort_values(["official_split", "image_id"], na_position="last")
        labels = labels.reset_index(drop=True)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    return LabelInventory(labels=labels, issues=issue_frame, csv_files=csv_files)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Replace a derived CSV only after a complete sibling file has been serialized."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict, destination: Path) -> None:
    """Atomically replace a derived JSON document after complete serialization."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
