"""Read-only BTXRD metadata audit and subtype migration artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from bone_cancer_ai.data.dedupe import (
    build_duplicate_groups,
    find_duplicate_edges,
    summarize_leakage,
)
from bone_cancer_ai.data.indexer import inspect_image
from bone_cancer_ai.data.samplers import split_generation_id
from bone_cancer_ai.data.schema import atomic_write_csv, atomic_write_json, opaque_id

BTXRD_SUBTYPE_COLUMNS = (
    "osteochondroma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "osteofibroma",
    "synovial osteochondroma",
    "other bt",
    "osteosarcoma",
    "other mt",
)
BTXRD_REQUIRED_COLUMNS = ("image_id", "tumor", *BTXRD_SUBTYPE_COLUMNS)
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class BtxrdAuditResult:
    """Validated BTXRD rows, duplicate evidence and machine-readable summary."""

    manifest: pd.DataFrame
    conflicts: pd.DataFrame
    duplicates: pd.DataFrame
    summary: dict[str, Any]


def _safe_image_path(images_root: Path, image_id: str) -> Path:
    candidate = Path(str(image_id).strip())
    if not candidate.name or candidate.name != str(image_id).strip() or candidate.is_absolute():
        raise ValueError(f"BTXRD image_id must be a filename, got {image_id!r}")
    if candidate.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported BTXRD image extension: {image_id!r}")
    root = images_root.resolve()
    resolved = (root / candidate).resolve()
    if root not in resolved.parents:
        raise ValueError(f"BTXRD image escapes images directory: {image_id!r}")
    return resolved


def load_btxrd_metadata(source_root: Path) -> pd.DataFrame:
    """Read and validate the BTXRD workbook without modifying the source."""
    source_root = Path(source_root)
    workbook = source_root / "dataset.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(f"BTXRD workbook not found: {workbook}")
    frame = pd.read_excel(workbook)
    missing = set(BTXRD_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required BTXRD columns: {sorted(missing)}")
    frame = frame.loc[:, list(BTXRD_REQUIRED_COLUMNS)].copy()
    frame["image_id"] = frame["image_id"].astype("string").str.strip()
    if frame["image_id"].isna().any() or frame["image_id"].eq("").any():
        raise ValueError("BTXRD image_id contains blank values")
    if frame["image_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["image_id"].duplicated(False), "image_id"].unique())
        raise ValueError(f"BTXRD image_id values are duplicated: {duplicates[:5]}")
    for column in ("tumor", *BTXRD_SUBTYPE_COLUMNS):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or ~numeric.isin([0, 1]).all():
            raise ValueError(f"BTXRD column {column!r} must contain only 0/1 values")
        frame[column] = numeric.astype("int8")
    return frame.sort_values("image_id").reset_index(drop=True)


def _metadata_targets(metadata: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    binary = metadata["tumor"].astype(int)
    active = metadata.loc[:, list(BTXRD_SUBTYPE_COLUMNS)].sum(axis=1)
    subtype = pd.Series("", index=metadata.index, dtype="string")
    conflict = pd.Series("", index=metadata.index, dtype="string")
    valid = binary.eq(1) & active.eq(1)
    for index in metadata.index[valid]:
        subtype.loc[index] = next(
            name for name in BTXRD_SUBTYPE_COLUMNS if int(metadata.at[index, name]) == 1
        )
    conflict_mask = binary.eq(1) & ~active.eq(1)
    conflict.loc[conflict_mask & active.eq(0)] = "missing_subtype"
    conflict.loc[conflict_mask & active.gt(1)] = "multiple_subtypes"
    normal_with_subtype = binary.eq(0) & active.gt(0)
    conflict.loc[normal_with_subtype] = "subtype_on_normal"
    return binary, subtype, valid, conflict


def _preferred_main_split(record_ids: str, main_official_splits: dict[str, list[str]]) -> str:
    """Resolve duplicate Main links with held-out partitions taking priority."""
    splits = {
        split
        for record_id in str(record_ids).split(";")
        for split in main_official_splits.get(record_id, [])
    }
    for split in ("test", "val", "train"):
        if split in splits:
            return split
    return ""


def _main_linkage(
    main_manifest_path: Path | None,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, str],
]:
    if main_manifest_path is None or not Path(main_manifest_path).is_file():
        return {}, {}, {}, {}
    main = pd.read_csv(main_manifest_path, dtype="string")
    required = {"record_id", "sha256"}
    if not required.issubset(main.columns):
        raise ValueError(
            f"Main manifest missing required columns: {sorted(required - set(main.columns))}"
        )
    exact: dict[str, list[str]] = {}
    paths: dict[str, str] = {}
    for row in main[["record_id", "sha256"]].dropna().itertuples(index=False):
        digest = str(row.sha256).strip().lower()
        if digest:
            exact.setdefault(digest, []).append(str(row.record_id))
    if "absolute_path" in main.columns:
        for row in main[["sha256", "absolute_path"]].dropna().itertuples(index=False):
            digest = str(row.sha256).strip().lower()
            path = str(row.absolute_path).strip()
            if digest and path:
                paths.setdefault(digest, path)
    official_splits: dict[str, list[str]] = {}
    for record_ids in exact.values():
        for record_id in record_ids:
            split = str(record_id).split("/", 1)[0]
            if split in {"train", "val", "test"}:
                official_splits.setdefault(str(record_id), []).append(split)
    phashes: dict[str, list[str]] = {}
    if "phash" in main.columns:
        for row in main[["record_id", "phash"]].dropna().itertuples(index=False):
            value = str(row.phash).strip().lower()
            if value:
                phashes.setdefault(value, []).append(str(row.record_id))
    return exact, phashes, official_splits, paths


def _stable_split_assignment(groups: list[str], seed: int) -> dict[str, str]:
    """Assign whole duplicate groups to deterministic fit/calibration/val/test bins."""
    if not groups:
        return {}
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    total = len(ordered)
    test_count = max(1, round(total * 0.10)) if total >= 2 else 0
    val_count = max(1, round(total * 0.10)) if total >= 3 else 0
    calibration_count = max(1, round(total * 0.10)) if total >= 4 else 0
    while test_count + val_count + calibration_count >= total and total > 1:
        if calibration_count:
            calibration_count -= 1
        elif val_count:
            val_count -= 1
        else:
            test_count -= 1
    assignment: dict[str, str] = {}
    cursor = 0
    for split, count in (
        ("test", test_count),
        ("val", val_count),
        ("calibration", calibration_count),
    ):
        for group in ordered[cursor : cursor + count]:
            assignment[group] = split
        cursor += count
    for group in ordered[cursor:]:
        assignment[group] = "fit_train"
    return assignment


def _assign_splits(
    manifest: pd.DataFrame,
    main_official_splits: dict[str, list[str]],
    seed: int,
    calibration_fraction: float = 0.15,
) -> tuple[pd.Series, dict[str, int]]:
    """Preserve Main held-out partitions and split only Main training groups."""
    group_main: dict[str, set[str]] = {}
    for group, values in manifest.groupby("split_group_id", sort=True):
        splits: set[str] = set()
        for records in values["main_match_record_ids"]:
            for record_id in str(records).split(";"):
                if record_id:
                    splits.update(main_official_splits.get(record_id, []))
        group_main[str(group)] = splits
    assignment: dict[str, str] = {}
    train_groups: list[str] = []
    unknown_groups: list[str] = []
    conflicts = 0
    for group in sorted(manifest["split_group_id"].unique()):
        values = group_main.get(str(group), set())
        if len(values) > 1:
            conflicts += 1
        if "test" in values:
            assignment[str(group)] = "test"
        elif "val" in values:
            assignment[str(group)] = "val"
        elif "train" in values:
            train_groups.append(str(group))
        else:
            unknown_groups.append(str(group))
    ordered_train = sorted(
        train_groups,
        key=lambda value: hashlib.sha256(f"calibration:{seed}:{value}".encode()).hexdigest(),
    )
    calibration_count = int(round(len(ordered_train) * calibration_fraction))
    for group in ordered_train[:calibration_count]:
        assignment[group] = "calibration"
    for group in ordered_train[calibration_count:]:
        assignment[group] = "fit_train"
    assignment.update(_stable_split_assignment(unknown_groups, seed))
    return manifest["split_group_id"].map(assignment).fillna("fit_train"), {
        "main_split_conflict_groups": conflicts,
        "main_train_groups": len(train_groups),
        "unmatched_groups": len(unknown_groups),
        "calibration_groups": calibration_count,
    }


def audit_btxrd(
    source_root: Path,
    *,
    main_manifest_path: Path | None = None,
    seed: int = 42,
    phash_distance: int = 4,
    quality_thresholds: dict[str, Any] | None = None,
) -> BtxrdAuditResult:
    """Audit BTXRD and return derived rows; the source tree is never written."""
    source_root = Path(source_root).resolve()
    images_root = source_root / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"BTXRD images directory not found: {images_root}")
    metadata = load_btxrd_metadata(source_root)
    binary, subtype, valid, conflict = _metadata_targets(metadata)
    thresholds = quality_thresholds or {}
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    exact_map, main_phash_map, main_official_splits, main_paths = _main_linkage(main_manifest_path)
    for index, row in metadata.iterrows():
        image_id = str(row["image_id"])
        path = _safe_image_path(images_root, image_id)
        if not path.is_file():
            missing_images.append(image_id)
            continue
        inspected = inspect_image(
            path,
            "btxrd",
            int(binary.loc[index]),
            thresholds,
            image_id=image_id,
        )
        inspected.update(
            {
                "sample_id": opaque_id("btxrd", image_id),
                "btxrd_absolute_path": inspected["absolute_path"],
                "binary_semantic": "bone_tumour" if int(binary.loc[index]) else "normal",
                "subtype_label": str(subtype.loc[index]),
                "subtype_valid": bool(valid.loc[index]),
                "subtype_conflict": str(conflict.loc[index]),
                "main_match_record_ids": ";".join(
                    exact_map.get(str(inspected["sha256"]).lower(), [])
                ),
                "main_phash_match_record_ids": ";".join(
                    main_phash_map.get(str(inspected["phash"]).lower(), [])
                ),
            }
        )
        main_path = main_paths.get(str(inspected["sha256"]).lower())
        if main_path:
            inspected["absolute_path"] = main_path
        rows.append(inspected)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("BTXRD contains no readable image rows")
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    duplicate_input = manifest[
        [
            "record_id",
            "image_id",
            "official_split",
            "sha256",
            "phash",
            "canonical_source_id",
            "binary_label",
            "width",
            "height",
        ]
    ]
    duplicates = find_duplicate_edges(duplicate_input, max_phash_distance=phash_distance)
    grouped = build_duplicate_groups(duplicate_input, duplicates)
    group_lookup: dict[str, str] = {}
    if not grouped.empty:
        for group, group_frame in grouped.groupby("duplicate_group", sort=True):
            for record in set(group_frame["record_id"]) | set(group_frame["other_record_id"]):
                group_lookup[str(record)] = str(group)
    manifest["split_group_id"] = [
        group_lookup.get(
            str(record),
            (
                f"main_sha_{str(sha)[:16]}"
                if str(main_ids)
                else f"single_{str(sha)[:16] or str(record)}"
            ),
        )
        for record, sha, main_ids in zip(
            manifest["record_id"],
            manifest["sha256"],
            manifest["main_match_record_ids"],
            strict=True,
        )
    ]
    manifest["assigned_split"], split_summary = _assign_splits(manifest, main_official_splits, seed)
    manifest["main_official_split"] = manifest["main_match_record_ids"].map(
        lambda value: _preferred_main_split(value, main_official_splits)
    )
    exact_conflict_groups = set(
        grouped.loc[
            grouped["label_conflict"].astype(bool)
            & grouped["match_type"].astype(str).str.contains("exact_sha256", regex=False),
            "duplicate_group",
        ]
    )
    manifest["duplicate_label_conflict"] = manifest["split_group_id"].isin(exact_conflict_groups)
    manifest["is_eligible"] = manifest["quality_status"].eq("readable")
    manifest["is_eligible"] &= manifest["binary_label"].notna()
    manifest["is_eligible"] &= manifest["subtype_conflict"].eq("") | manifest["binary_label"].eq(0)
    manifest["is_eligible"] &= ~manifest["duplicate_label_conflict"]
    manifest["split_generation_id"] = ""
    eligible = manifest.loc[manifest["is_eligible"]].copy()
    manifest.loc[manifest["is_eligible"], "split_generation_id"] = _serialized_split_generation(
        eligible
    )
    conflicts = manifest.loc[
        manifest["subtype_conflict"].ne(""),
        ["record_id", "image_id", "binary_label", "subtype_conflict", "subtype_label"],
    ].copy()
    split_counts = {
        str(key): int(value) for key, value in manifest["assigned_split"].value_counts().items()
    }
    eligible_split_counts = {
        str(key): int(value)
        for key, value in manifest.loc[manifest["is_eligible"], "assigned_split"]
        .value_counts()
        .items()
    }
    exact_matches = manifest["main_match_record_ids"].ne("")
    summary: dict[str, Any] = {
        "source_root": str(source_root),
        "metadata_rows": int(len(metadata)),
        "image_files": int(len(list(images_root.iterdir()))),
        "audited_images": int(len(manifest)),
        "missing_images": sorted(missing_images),
        "extra_image_files": sorted(
            str(path.name)
            for path in images_root.iterdir()
            if path.is_file() and path.name not in set(metadata["image_id"])
        ),
        "binary_counts": {
            str(key): int(value)
            for key, value in manifest["binary_label"].value_counts().sort_index().items()
        },
        "subtype_valid_rows": int(manifest["subtype_valid"].sum()),
        "subtype_conflict_rows": int(manifest["subtype_conflict"].ne("").sum()),
        "duplicate_label_conflict_rows": int(manifest["duplicate_label_conflict"].sum()),
        "subtype_counts": {
            str(key): int(value)
            for key, value in manifest.loc[manifest["subtype_valid"], "subtype_label"]
            .value_counts()
            .items()
        },
        "exact_hash_match_rows": int(exact_matches.sum()),
        "exact_hash_unmatched_rows": int((~exact_matches).sum()),
        "main_phash_match_rows": int(manifest["main_phash_match_record_ids"].ne("").sum()),
        "duplicate_evidence": summarize_leakage(grouped),
        "split_counts": split_counts,
        "eligible_split_counts": eligible_split_counts,
        "split_group_count": int(manifest["split_group_id"].nunique()),
        "preserved_main_split_rows": int(manifest["main_match_record_ids"].ne("").sum()),
        "linked_to_main_path_rows": int(
            manifest["btxrd_absolute_path"].ne(manifest["absolute_path"]).sum()
        ),
        **split_summary,
        "source_write_policy": "read_only",
        "seed": int(seed),
        "phash_distance": int(phash_distance),
    }
    return BtxrdAuditResult(
        manifest=manifest, conflicts=conflicts, duplicates=grouped, summary=summary
    )


def _manifest_digest(manifest: pd.DataFrame) -> str:
    stable = manifest.sort_values("record_id").copy().astype("string").fillna("")
    payload = stable.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialized_split_generation(frame: pd.DataFrame) -> str:
    """Fingerprint the same CSV-normalized values that downstream loaders read."""
    serialized = frame.to_csv(index=False, lineterminator="\n")
    normalized = pd.read_csv(StringIO(serialized), keep_default_na=True)
    return split_generation_id(normalized)


def write_btxrd_artifacts(
    source_root: Path,
    *,
    main_manifest_path: Path | None = None,
    output_root: Path = Path("workspace/btxrd"),
    seed: int = 42,
    phash_distance: int = 4,
    quality_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write idempotent BTXRD audit artifacts outside both source datasets."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root)
    if source_root == output_root.resolve() or source_root in output_root.resolve().parents:
        raise ValueError("BTXRD output_root must not be inside the read-only source tree")
    result = audit_btxrd(
        source_root,
        main_manifest_path=main_manifest_path,
        seed=seed,
        phash_distance=phash_distance,
        quality_thresholds=quality_thresholds,
    )
    atomic_write_csv(result.manifest, output_root / "btxrd_manifest.csv")
    atomic_write_csv(result.conflicts, output_root / "subtype_conflicts.csv")
    atomic_write_csv(result.duplicates, output_root / "duplicates.csv")
    for split in ("fit_train", "calibration", "val", "test"):
        atomic_write_csv(
            result.manifest.loc[
                result.manifest["assigned_split"].eq(split)
                & result.manifest["is_eligible"].astype(bool)
            ].copy(),
            output_root / f"{split}.csv",
        )
    summary = dict(result.summary)
    summary["manifest_sha256"] = _manifest_digest(result.manifest)
    summary["derived_output_root"] = str(output_root.resolve())
    atomic_write_json(summary, output_root / "audit_summary.json")
    return summary
