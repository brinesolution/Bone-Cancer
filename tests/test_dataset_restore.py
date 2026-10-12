"""Portable, hash-bound restoration of the original combined Main dataset."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from scripts.dataset_restore import build_manifest, restore_main, validate_main


def _write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("image_id", "binary_label"))
        writer.writerows(rows)


def _fixture(root: Path) -> tuple[Path, Path]:
    btxrd = root / "source" / "BTXRD" / "images"
    roboflow = root / "source" / "Main"
    btxrd.mkdir(parents=True)
    # Same basename with different bytes proves that identity comes from hashes.
    (btxrd / "IMG000001.jpeg").write_bytes(b"btxrd pixels")
    rows = {
        "train": [("IMG000001.jpeg", 0), ("normal_a.jpg", 0)],
        "val": [("cancer_b.jpg", 1)],
        "test": [("normal_c.jpg", 0)],
    }
    for split, items in rows.items():
        folder = roboflow / split
        folder.mkdir(parents=True)
        for name, _label in items:
            payload = b"btxrd pixels" if name == "IMG000001.jpeg" else (split + name).encode()
            (folder / name).write_bytes(payload)
        _write_csv(folder / f"_{split}_labels.csv", items)
    contract = root / "contract.csv"
    metadata = root / "metadata.json"
    build_manifest(roboflow, btxrd.parent.parent, contract, metadata)
    return roboflow, btxrd.parent.parent


def test_restores_original_paths_labels_splits_and_bytes(tmp_path: Path):
    source, source_root = _fixture(tmp_path)
    destination = tmp_path / "clone" / "dataset" / "Main"
    restore_main(source_root, destination, tmp_path / "contract.csv", tmp_path / "metadata.json")
    assert validate_main(destination, tmp_path / "contract.csv", tmp_path / "metadata.json")[
        "valid"
    ]
    for split in ("train", "val", "test"):
        assert (destination / split / f"_{split}_labels.csv").read_bytes() == (
            source / split / f"_{split}_labels.csv"
        ).read_bytes()
    assert (destination / "train/IMG000001.jpeg").read_bytes() == b"btxrd pixels"


def test_source_export_split_names_do_not_change_original_main_assignments(tmp_path: Path):
    source, source_root = _fixture(tmp_path)
    source_image = source / "train/normal_a.jpg"
    moved = source / "test/normal_a.jpg"
    source_image.replace(moved)
    destination = tmp_path / "clone/Main"
    restore_main(source_root, destination, tmp_path / "contract.csv", tmp_path / "metadata.json")
    assert (destination / "train/normal_a.jpg").read_bytes() == b"trainnormal_a.jpg"
    assert not (destination / "test/normal_a.jpg").exists()


def test_partial_restore_resumes_and_never_overwrites_different_bytes(tmp_path: Path):
    _fixture(tmp_path)
    destination = tmp_path / "clone/Main"
    destination.mkdir(parents=True)
    (destination / "train/normal_a.jpg").parent.mkdir()
    (destination / "train/normal_a.jpg").write_bytes(b"wrong existing pixels")
    with pytest.raises(ValueError, match="preserved"):
        restore_main(
            tmp_path / "source", destination, tmp_path / "contract.csv", tmp_path / "metadata.json"
        )
    assert (destination / "train/normal_a.jpg").read_bytes() == b"wrong existing pixels"


def test_existing_valid_main_files_need_no_source_export(tmp_path: Path):
    source, _source_root = _fixture(tmp_path)
    destination = tmp_path / "clone/Main"
    for split in ("train", "val", "test"):
        (destination / split).mkdir(parents=True)
        for item in (source / split).iterdir():
            (destination / split / item.name).write_bytes(item.read_bytes())
    result = restore_main([], destination, tmp_path / "contract.csv", tmp_path / "metadata.json")
    assert result["total_images"] == 4


def test_partial_main_repairs_btxrd_image_without_roboflow_source(tmp_path: Path):
    source, source_root = _fixture(tmp_path)
    destination = tmp_path / "clone/Main"
    for split in ("train", "val", "test"):
        (destination / split).mkdir(parents=True)
        for item in (source / split).iterdir():
            (destination / split / item.name).write_bytes(item.read_bytes())
    target = destination / "train/IMG000001.jpeg"
    target.unlink()

    btxrd_only = tmp_path / "btxrd-only"
    image_dir = btxrd_only / "BTXRD/images"
    image_dir.mkdir(parents=True)
    (image_dir / "IMG000001.jpeg").write_bytes(
        (source_root / "BTXRD/images/IMG000001.jpeg").read_bytes()
    )
    result = restore_main(
        [btxrd_only], destination, tmp_path / "contract.csv", tmp_path / "metadata.json"
    )
    assert result["total_images"] == 4
    assert target.read_bytes() == b"btxrd pixels"
    for split in ("train", "val", "test"):
        assert (destination / split / f"_{split}_labels.csv").read_bytes() == (
            source / split / f"_{split}_labels.csv"
        ).read_bytes()


def test_rejects_tampered_source_and_wrong_class_label(tmp_path: Path):
    _fixture(tmp_path)
    image = tmp_path / "source/Main/train/normal_a.jpg"
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        restore_main(
            tmp_path / "source",
            tmp_path / "clone/Main",
            tmp_path / "contract.csv",
            tmp_path / "metadata.json",
        )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["class_names"] == {"0": "normal", "1": "cancer"}


def test_manifest_rows_bind_hash_source_split_label_and_csv_order(tmp_path: Path):
    _fixture(tmp_path)
    result = validate_main(
        tmp_path / "source/Main", tmp_path / "contract.csv", tmp_path / "metadata.json"
    )
    assert result["valid"]
    assert result["counts"] == {"train": 2, "val": 1, "test": 1}
    assert result["source_counts"] == {"btxrd": 1, "roboflow_candidate": 3}


def test_rejects_archive_paths_and_contract_path_escape(tmp_path: Path):
    source, _ = _fixture(tmp_path)
    row_path = tmp_path / "contract.csv"
    text = row_path.read_text()
    row_path.write_text(text.replace("train/normal_a.jpg", "../outside.jpg"))
    with pytest.raises(ValueError, match="checksum"):
        validate_main(source, row_path, tmp_path / "metadata.json")


def test_checked_in_contract_validates_current_full_main_tree():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "dataset/Main").is_dir():
        pytest.skip("the ignored local Main dataset is unavailable")
    result = validate_main(
        repo / "dataset/Main",
        repo / "configs/main_dataset_manifest.csv",
        repo / "configs/main_dataset_manifest.json",
    )
    assert result["total_images"] == 12_557
    assert result["counts"] == {"train": 8_789, "val": 1_884, "test": 1_884}
    assert result["source_counts"] == {"btxrd": 3_746, "roboflow_candidate": 8_811}


def test_contract_refuses_a_reparse_point_source_root(tmp_path: Path):
    _fixture(tmp_path)
    link = tmp_path / "source-link"
    result = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(link), str(tmp_path / "source")],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip("Windows junction creation is unavailable")
    with pytest.raises(ValueError, match="link"):
        validate_main(link / "Main", tmp_path / "contract.csv", tmp_path / "metadata.json")


@pytest.mark.parametrize("target", ["manifest", "metadata"])
def test_checked_in_contract_has_code_pinned_integrity(tmp_path: Path, target: str):
    repo = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "main_dataset_manifest.csv"
    metadata = tmp_path / "main_dataset_manifest.json"
    manifest.write_bytes((repo / "configs/main_dataset_manifest.csv").read_bytes())
    metadata.write_bytes((repo / "configs/main_dataset_manifest.json").read_bytes())
    if target == "manifest":
        manifest.write_bytes(manifest.read_bytes() + b"tampered")
    else:
        metadata.write_text(metadata.read_text().replace('"schema": 1', '"schema": 2'))
    with pytest.raises(ValueError, match="Pinned Main dataset"):
        validate_main(repo / "dataset/Main", manifest, metadata)
