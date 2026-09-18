from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from bone_cancer_ai.data.btxrd_migration import (
    BTXRD_SUBTYPE_COLUMNS,
    audit_btxrd,
    write_btxrd_artifacts,
)


def _write_image(path: Path, value: int) -> None:
    image = Image.new("L", (24, 16), value)
    image.save(path, format="JPEG")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "BTXRD"
    images = source / "images"
    images.mkdir(parents=True)
    _write_image(images / "IMG000001.jpeg", 20)
    _write_image(images / "IMG000002.jpeg", 180)
    (images / "IMG000003.jpeg").write_bytes((images / "IMG000002.jpeg").read_bytes())
    rows = []
    for image_id, tumor, subtype in [
        ("IMG000001.jpeg", 0, ""),
        ("IMG000002.jpeg", 1, "osteosarcoma"),
        ("IMG000003.jpeg", 1, "conflict"),
    ]:
        row = {column: 0 for column in BTXRD_SUBTYPE_COLUMNS}
        if subtype == "osteosarcoma":
            row["osteosarcoma"] = 1
        if subtype == "conflict":
            row["osteosarcoma"] = 1
            row["other mt"] = 1
        rows.append({"image_id": image_id, "tumor": tumor, **row})
    pd.DataFrame(rows).to_excel(source / "dataset.xlsx", index=False)
    main = tmp_path / "main.csv"
    main_sha = hashlib.sha256((images / "IMG000002.jpeg").read_bytes()).hexdigest()
    pd.DataFrame(
        [{"record_id": "train/main-match.jpeg", "image_id": "main-match.jpeg", "sha256": main_sha}]
    ).to_csv(main, index=False)
    return source, main, images


def test_audit_masks_conflicting_subtype_and_links_exact_hash(tmp_path: Path):
    source, main, _ = _fixture(tmp_path)
    result = audit_btxrd(source, main_manifest_path=main, seed=7, phash_distance=0)
    assert result.summary["metadata_rows"] == 3
    assert result.summary["image_files"] == 3
    assert result.summary["subtype_valid_rows"] == 1
    assert result.summary["subtype_conflict_rows"] == 1
    assert result.summary["exact_hash_match_rows"] == 2
    assert (
        result.manifest.loc[result.manifest.image_id == "IMG000001.jpeg", "subtype_label"].iloc[0]
        == ""
    )
    assert (
        result.manifest.loc[result.manifest.image_id == "IMG000002.jpeg", "subtype_label"].iloc[0]
        == "osteosarcoma"
    )
    assert (
        result.manifest.loc[result.manifest.image_id == "IMG000003.jpeg", "subtype_label"].iloc[0]
        == ""
    )
    assert result.manifest["assigned_split"].notna().all()
    assert result.manifest.groupby("split_group_id")["assigned_split"].nunique().max() == 1
    assert (
        result.manifest.loc[
            result.manifest.image_id == "IMG000002.jpeg", "main_official_split"
        ].iloc[0]
        == "train"
    )


def test_audit_is_deterministic_and_writes_only_derived_artifacts(tmp_path: Path):
    source, main, images = _fixture(tmp_path)
    before = {
        p.relative_to(source).as_posix(): p.stat().st_mtime_ns
        for p in source.rglob("*")
        if p.is_file()
    }
    first = write_btxrd_artifacts(
        source, main_manifest_path=main, output_root=tmp_path / "out", seed=7, phash_distance=0
    )
    second = write_btxrd_artifacts(
        source, main_manifest_path=main, output_root=tmp_path / "out", seed=7, phash_distance=0
    )
    after = {
        p.relative_to(source).as_posix(): p.stat().st_mtime_ns
        for p in source.rglob("*")
        if p.is_file()
    }
    assert before == after
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (tmp_path / "out" / "btxrd_manifest.csv").exists()
    assert (tmp_path / "out" / "audit_summary.json").exists()
    assert (tmp_path / "out" / "duplicates.csv").exists()


def test_missing_required_column_fails_closed(tmp_path: Path):
    source, _, _ = _fixture(tmp_path)
    pd.DataFrame([{"image_id": "IMG000001.jpeg"}]).to_excel(source / "dataset.xlsx", index=False)
    with pytest.raises(ValueError, match="required BTXRD columns"):
        audit_btxrd(source)
