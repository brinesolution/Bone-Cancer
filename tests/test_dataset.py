"""Dataset indexing contracts use tiny synthetic files and never mutate the source dataset."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def write_labels(folder: Path, split: str, rows: list[tuple[str, object]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", "binary_label"]).to_csv(
        folder / f"_{split}_labels.csv", index=False
    )


def test_canonical_source_id_removes_only_export_suffixes():
    from bone_cancer_ai.data.schema import canonical_source_id, record_id

    assert canonical_source_id("Case_A_png.rf.0123456789abcdef0123456789abcdef.jpg") == "case_a"
    assert canonical_source_id("tumour.rf.short.jpg") == "tumour.rf.short"
    assert canonical_source_id("patient_augmented.jpeg") == "patient_augmented"
    assert record_id("val", "A B.jpg") == "val/A B.jpg"


def test_load_main_labels_maps_valid_rows(tmp_path):
    from bone_cancer_ai.data.schema import load_main_labels

    write_labels(tmp_path / "train", "train", [("a.jpg", 0), ("b.jpg", 1)])
    write_labels(tmp_path / "val", "val", [("c.jpg", 0)])
    inventory = load_main_labels(tmp_path, ["train", "val"])
    assert inventory.labels[["record_id", "binary_label"]].to_dict("records") == [
        {"record_id": "train/a.jpg", "binary_label": 0},
        {"record_id": "train/b.jpg", "binary_label": 1},
        {"record_id": "val/c.jpg", "binary_label": 0},
    ]
    assert inventory.issues.empty


@pytest.mark.parametrize(
    ("mutate", "issue"),
    [
        (lambda frame: frame.assign(extra="x"), "unexpected_label_columns"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate_label_id",
        ),
        (lambda frame: frame.assign(image_id=["", "b.jpg"]), "missing_image_id"),
        (lambda frame: frame.assign(binary_label=["x", 1]), "invalid_binary_label"),
        (lambda frame: frame.assign(binary_label=[2, 1]), "invalid_binary_label"),
    ],
)
def test_load_main_labels_reports_structural_issues(tmp_path, mutate, issue):
    from bone_cancer_ai.data.schema import load_main_labels

    split = tmp_path / "train"
    write_labels(split, "train", [("a.jpg", 0), ("b.jpg", 1)])
    path = split / "_train_labels.csv"
    mutate(pd.read_csv(path)).to_csv(path, index=False)
    inventory = load_main_labels(tmp_path, ["train"])
    assert issue in set(inventory.issues["issue_type"])


def test_load_main_labels_requires_exactly_one_csv(tmp_path):
    from bone_cancer_ai.data.schema import load_main_labels

    (tmp_path / "train").mkdir()
    missing = load_main_labels(tmp_path, ["train"])
    assert missing.issues["issue_type"].tolist() == ["missing_label_csv"]
    write_labels(tmp_path / "train", "train", [("a.jpg", 0)])
    write_labels(tmp_path / "train", "other", [("a.jpg", 0)])
    multiple = load_main_labels(tmp_path, ["train"])
    assert multiple.issues["issue_type"].tolist() == ["multiple_label_csv"]


def test_atomic_write_csv_replaces_complete_table(tmp_path):
    from bone_cancer_ai.data.schema import atomic_write_csv

    path = tmp_path / "nested" / "table.csv"
    atomic_write_csv(pd.DataFrame({"value": [1, 2]}), path)
    assert pd.read_csv(path)["value"].tolist() == [1, 2]
    atomic_write_csv(pd.DataFrame({"value": [3]}), path)
    assert pd.read_csv(path)["value"].tolist() == [3]
    assert not list(path.parent.glob("*.tmp"))


def test_inspect_image_records_grayscale_and_quality(tmp_path):
    from bone_cancer_ai.data.indexer import inspect_image

    image_path = tmp_path / "Sample_png.rf.0123456789abcdef0123456789abcdef.jpg"
    image = Image.fromarray(np.arange(96, dtype=np.uint8).reshape(8, 12))
    image.save(image_path, quality=100)
    row = inspect_image(image_path, "train", 1, {"thumbnail_size": 64})
    assert row["record_id"] == f"train/{image_path.name}"
    assert row["width"] == 12 and row["height"] == 8 and row["channels"] == 1
    assert row["pil_mode"] == "L" and row["binary_label"] == 1
    assert row["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert len(row["phash"]) == 16
    assert row["quality_status"] == "readable"
    assert row["decode_error"] == ""


def test_inspect_image_detects_rgb_encoded_grayscale_and_blank(tmp_path):
    from bone_cancer_ai.data.indexer import inspect_image

    image_path = tmp_path / "blank.png"
    Image.new("RGB", (9, 7), (20, 20, 20)).save(image_path)
    row = inspect_image(
        image_path,
        "val",
        0,
        {"thumbnail_size": 64, "blank_std_threshold": 0.5, "near_blank_std_threshold": 3},
    )
    assert row["channels"] == 3 and row["is_rgb_encoded_grayscale"] is True
    assert row["quality_status"] == "blank"


def test_inspect_image_contains_decode_error(tmp_path):
    from bone_cancer_ai.data.indexer import inspect_image

    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    row = inspect_image(path, "test", None, {})
    assert row["quality_status"] == "unreadable"
    assert row["width"] is None and row["phash"] == ""
    assert "UnidentifiedImageError" in row["decode_error"]


def test_index_dataset_reports_missing_and_extra_label_rows(tmp_path):
    from bone_cancer_ai.data.indexer import index_dataset

    root = tmp_path / "source"
    write_labels(root / "train", "train", [("present.png", 0), ("missing.png", 1)])
    Image.new("L", (8, 8), 100).save(root / "train" / "present.png")
    Image.new("L", (8, 8), 150).save(root / "train" / "extra.png")
    config = {
        "dataset_root": str(root),
        "official_splits": ["train"],
        "image_extensions": [".png"],
        "audit": {"workers": 1, "quality": {"thumbnail_size": 32}},
    }
    master, issues, summary = index_dataset(config)
    assert master["image_id"].tolist() == ["extra.png", "present.png"]
    assert pd.isna(master.loc[master.image_id.eq("extra.png"), "binary_label"]).all()
    assert set(issues["issue_type"]) == {"image_missing_label", "label_missing_image"}
    assert summary["images_indexed"] == 2


def test_index_cli_writes_derived_outputs_outside_source(tmp_path):
    project = tmp_path / "project"
    source = project / "dataset" / "Main"
    write_labels(source / "train", "train", [("a.png", 0)])
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(source / "train/a.png")
    config = project / "data.yaml"
    config.write_text(
        "\n".join(
            [
                f'project_root: "{project.as_posix()}"',
                f'dataset_root: "{source.as_posix()}"',
                "source_read_only: true",
                "official_splits: [train]",
                "image_extensions: [.png]",
                "audit:",
                "  workers: 1",
                "  quality:",
                "    thumbnail_size: 32",
            ]
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts/01_index_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    master = pd.read_csv(project / "workspace/manifests/dataset_master.csv")
    assert master["record_id"].tolist() == ["train/a.png"]
    quality = pd.read_csv(project / "workspace/manifests/quality_report.csv")
    assert quality["record_id"].tolist() == ["train/a.png"]
    summary = json.loads((project / "logs/dataset_index_summary.json").read_text())
    assert summary["images_indexed"] == 1
    assert sorted(path.name for path in (source / "train").iterdir()) == [
        "_train_labels.csv",
        "a.png",
    ]


def test_local_provenance_loads_binary_risk_subtype_and_spatial_evidence(tmp_path):
    from bone_cancer_ai.data.provenance import load_local_provenance

    binary = tmp_path / "binary"
    (binary / "train/cancer").mkdir(parents=True)
    (binary / "train/cancer/Case_A_png.rf.0123456789abcdef0123456789abcdef.jpg").touch()
    three = tmp_path / "three"
    (three / "train/benign").mkdir(parents=True)
    (three / "train/benign/IMG000002.jpeg").touch()
    btxrd = tmp_path / "btxrd"
    (btxrd / "Annotations").mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "image_id": ["IMG000002.jpeg"],
            "tumor": [1],
            "benign": [1],
            "malignant": [0],
            "osteochondroma": [1],
            "other mt": [0],
        }
    )
    frame.to_excel(btxrd / "dataset.xlsx", index=False)
    (btxrd / "Annotations/IMG000002.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {"shape_type": "rectangle", "label": "osteochondroma"},
                    {"shape_type": "polygon", "label": "osteochondroma"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = {
        "provenance": {
            "roboflow_binary_root": "binary",
            "three_class_root": "three",
            "btxrd_root": "btxrd",
        }
    }
    result = load_local_provenance(config, tmp_path)
    case = result[result.canonical_source_id.eq("case_a")].iloc[0]
    assert case.binary_semantic == "bone_tumour"
    image = result[
        result.canonical_source_id.eq("img000002") & result.source_name.eq("btxrd_workbook")
    ].iloc[0]
    assert image.risk_label == "benign"
    assert image.subtype_label == "osteochondroma"
    assert bool(image.has_box) is True and bool(image.has_mask) is True


def test_audit_labels_enriches_verified_rows_and_reports_conflicts():
    from bone_cancer_ai.data.audit import audit_labels

    master = pd.DataFrame(
        [
            {
                "record_id": "train/a.jpg",
                "image_id": "a.jpg",
                "official_split": "train",
                "canonical_source_id": "a",
                "sha256": "sha-a",
                "binary_label": 1,
                "binary_semantic": "",
                "subtype_label": "",
                "risk_label": "",
                "has_box": False,
                "has_mask": False,
            },
            {
                "record_id": "val/b.jpg",
                "image_id": "b.jpg",
                "official_split": "val",
                "canonical_source_id": "b",
                "sha256": "sha-b",
                "binary_label": 0,
                "binary_semantic": "",
                "subtype_label": "",
                "risk_label": "",
                "has_box": False,
                "has_mask": False,
            },
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "canonical_source_id": "a",
                "source_name": "folders",
                "source_sha256": "sha-a",
                "binary_semantic": "bone_tumour",
                "risk_label": "malignant",
                "subtype_label": "osteosarcoma",
                "has_box": True,
                "has_mask": True,
            },
            {
                "canonical_source_id": "b",
                "source_name": "folders",
                "source_sha256": "sha-b",
                "binary_semantic": "bone_tumour",
                "risk_label": "",
                "subtype_label": "",
                "has_box": False,
                "has_mask": False,
            },
        ]
    )
    enriched, issues, summary = audit_labels(master, provenance)
    row = enriched.loc[enriched.record_id.eq("train/a.jpg")].iloc[0]
    assert row.binary_semantic == "bone_tumour"
    assert row.risk_label == "malignant" and row.subtype_label == "osteosarcoma"
    assert bool(row.has_box) is True and bool(row.has_mask) is True
    conflict = issues.loc[issues.record_id.eq("val/b.jpg")]
    assert conflict.issue_type.tolist() == ["binary_semantic_conflict"]
    assert summary["provenance_matched_images"] == 2
    assert summary["binary_semantics_verified"] is False


def test_audit_labels_verifies_complete_consistent_binary_semantics():
    from bone_cancer_ai.data.audit import audit_labels

    master = pd.DataFrame(
        [
            {
                "record_id": "train/a.jpg",
                "image_id": "a.jpg",
                "official_split": "train",
                "canonical_source_id": "a",
                "sha256": "sha-a",
                "binary_label": 1,
                "binary_semantic": "",
                "subtype_label": "",
                "risk_label": "",
                "has_box": False,
                "has_mask": False,
            },
            {
                "record_id": "val/b.jpg",
                "image_id": "b.jpg",
                "official_split": "val",
                "canonical_source_id": "b",
                "sha256": "sha-b",
                "binary_label": 0,
                "binary_semantic": "",
                "subtype_label": "",
                "risk_label": "",
                "has_box": False,
                "has_mask": False,
            },
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "canonical_source_id": "a",
                "source_name": "source",
                "source_sha256": "sha-a",
                "binary_semantic": "bone_tumour",
                "risk_label": "",
                "subtype_label": "",
                "has_box": False,
                "has_mask": False,
            },
            {
                "canonical_source_id": "b",
                "source_name": "source",
                "source_sha256": "sha-b",
                "binary_semantic": "normal",
                "risk_label": "",
                "subtype_label": "",
                "has_box": False,
                "has_mask": False,
            },
        ]
    )
    enriched, issues, summary = audit_labels(master, provenance)
    assert issues.empty
    assert enriched.binary_semantic.tolist() == ["bone_tumour", "normal"]
    assert summary["binary_semantics_verified"] is True
    assert summary["binary_class_names"] == {"0": "normal", "1": "bone_tumour"}


def test_audit_requires_content_anchored_provenance():
    from bone_cancer_ai.data.audit import audit_labels

    master = pd.DataFrame(
        [
            {
                "record_id": "train/a.jpg",
                "image_id": "a.jpg",
                "official_split": "train",
                "canonical_source_id": "a",
                "sha256": "main-sha",
                "binary_label": 1,
            }
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "canonical_source_id": "a",
                "source_name": "metadata",
                "source_sha256": "different-sha",
                "binary_semantic": "bone_tumour",
                "risk_label": "malignant",
                "subtype_label": "osteosarcoma",
                "has_box": True,
                "has_mask": True,
                "source_issue": "",
            }
        ]
    )
    enriched, issues, summary = audit_labels(master, provenance)
    assert issues.issue_type.tolist() == ["unverified_provenance_content"]
    assert enriched.binary_semantic.tolist() == [""]
    assert enriched.subtype_label.tolist() == [""]
    assert enriched.has_box.tolist() == [False]
    assert summary["binary_semantics_verified"] is False


def test_audit_uses_only_content_anchored_rows_within_a_canonical_group():
    from bone_cancer_ai.data.audit import audit_labels

    master = pd.DataFrame(
        [
            {
                "record_id": "train/a.jpg",
                "image_id": "a.jpg",
                "official_split": "train",
                "canonical_source_id": "a",
                "sha256": "main-sha",
                "binary_label": 1,
            }
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "canonical_source_id": "a",
                "source_name": "exact_metadata",
                "source_sha256": "main-sha",
                "binary_semantic": "bone_tumour",
                "risk_label": "benign",
                "subtype_label": "osteochondroma",
                "has_box": True,
                "has_mask": True,
                "source_issue": "",
            },
            {
                "canonical_source_id": "a",
                "source_name": "transformed_export",
                "source_sha256": "different-sha",
                "binary_semantic": "normal",
                "risk_label": "malignant",
                "subtype_label": "osteosarcoma",
                "has_box": False,
                "has_mask": False,
                "source_issue": "",
            },
        ]
    )
    enriched, issues, summary = audit_labels(master, provenance)
    assert issues.empty
    assert enriched.binary_semantic.tolist() == ["bone_tumour"]
    assert enriched.risk_label.tolist() == ["benign"]
    assert enriched.subtype_label.tolist() == ["osteochondroma"]
    assert summary["nonexact_provenance_rows"] == 1


def test_repeated_audit_resets_stale_enrichment_and_replaces_provenance_issues():
    from bone_cancer_ai.data.audit import audit_labels

    master = pd.DataFrame(
        [
            {
                "record_id": "train/a.jpg",
                "image_id": "a.jpg",
                "official_split": "train",
                "canonical_source_id": "a",
                "sha256": "sha-a",
                "binary_label": 1,
            }
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "canonical_source_id": "a",
                "source_name": "metadata",
                "source_sha256": "sha-a",
                "binary_semantic": "bone_tumour",
                "risk_label": "benign",
                "subtype_label": "osteochondroma",
                "has_box": True,
                "has_mask": True,
                "source_issue": "subtype_one_hot_count=2",
            }
        ]
    )
    enriched, first_issues, _ = audit_labels(master, provenance)
    assert enriched.subtype_label.tolist() == [""]
    empty_provenance = provenance.iloc[0:0]
    second, second_issues, _ = audit_labels(enriched, empty_provenance, first_issues)
    assert second.binary_semantic.tolist() == [""]
    assert second.risk_label.tolist() == [""]
    assert second.has_box.tolist() == [False]
    assert second_issues.issue_type.tolist() == ["missing_provenance_evidence"]


def test_save_audit_figures_creates_three_readable_pngs(tmp_path):
    from bone_cancer_ai.data.audit import save_audit_figures

    paths = []
    rows = []
    for index, (split, label) in enumerate([("train", 0), ("train", 1), ("val", 0)]):
        path = tmp_path / f"source-{index}.png"
        Image.fromarray(np.arange(120, dtype=np.uint8).reshape(10, 12) + index).save(path)
        paths.append(path)
        rows.append(
            {
                "record_id": f"{split}/item-{index}.png",
                "absolute_path": str(path),
                "official_split": split,
                "binary_label": label,
                "binary_semantic": "normal" if label == 0 else "bone_tumour",
                "width": 12,
                "height": 10,
                "quality_status": "readable",
            }
        )
    outputs = save_audit_figures(pd.DataFrame(rows), tmp_path / "figures", seed=42)
    assert {path.name for path in outputs} == {
        "dataset_class_distribution.png",
        "dimension_distribution.png",
        "sample_grid.png",
    }
    for path in outputs:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 600 and image.height >= 400


def test_audit_cli_enriches_manifest_and_writes_figures(tmp_path):
    project = tmp_path / "project"
    source = project / "dataset/Main"
    write_labels(source / "train", "train", [("a.png", 0)])
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(source / "train/a.png")
    provenance = project / "provenance/train/normal"
    provenance.mkdir(parents=True)
    (provenance / "a.png").write_bytes((source / "train/a.png").read_bytes())
    config = project / "data.yaml"
    config.write_text(
        "\n".join(
            [
                f'project_root: "{project.as_posix()}"',
                f'dataset_root: "{source.as_posix()}"',
                "source_read_only: true",
                "official_splits: [train]",
                "image_extensions: [.png]",
                "seed: 42",
                "audit:",
                "  workers: 1",
                "  sample_grid_size: 1",
                "  quality:",
                "    thumbnail_size: 32",
                "provenance:",
                "  roboflow_binary_root: provenance",
            ]
        ),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[1]
    index = subprocess.run(
        [sys.executable, str(repository / "scripts/01_index_dataset.py"), "--config", str(config)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert index.returncode == 0, index.stderr + index.stdout
    audit = subprocess.run(
        [sys.executable, str(repository / "scripts/02_audit_dataset.py"), "--config", str(config)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert audit.returncode == 0, audit.stderr + audit.stdout
    master = pd.read_csv(project / "workspace/manifests/dataset_master.csv")
    assert master["binary_semantic"].tolist() == ["normal"]
    summary = json.loads((project / "logs/dataset_audit_summary.json").read_text())
    assert summary["binary_semantics_verified"] is True
    assert summary["dimension_summary"] == {
        "width": {"min": 8, "median": 8.0, "max": 8},
        "height": {"min": 8, "median": 8.0, "max": 8},
        "unique_width_height_pairs": 1,
    }
    assert summary["class_counts_by_split"] == {"train": {"0": 1}}
    assert {path.name for path in (project / "artifacts/figures").glob("*.png")} == {
        "dataset_class_distribution.png",
        "dimension_distribution.png",
        "sample_grid.png",
    }
