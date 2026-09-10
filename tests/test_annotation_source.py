from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, label: str = "osteosarcoma") -> tuple[Path, dict[str, Path]]:
    project = tmp_path / "project"
    main = project / "dataset" / "Main" / "train"
    source_images = project / "dataset" / "BTXRD" / "images"
    source_json = project / "dataset" / "BTXRD" / "Annotations"
    annotation_root = project / "workspace" / "annotations"
    manifest_dir = project / "workspace" / "manifests"
    for directory in (main, source_images, source_json, annotation_root, manifest_dir):
        directory.mkdir(parents=True)

    source = source_images / "IMG000001.jpeg"
    Image.new("RGB", (8, 6), (20, 30, 40)).save(source)
    main_image = main / "opaque-main.jpeg"
    main_image.write_bytes(source.read_bytes())
    payload = {
        "imagePath": source.name,
        "imageWidth": 8,
        "imageHeight": 6,
        "shapes": [
            {"label": label, "shape_type": "rectangle", "points": [[1, 1], [5, 4]]},
            {"label": "lesion", "shape_type": "polygon", "points": [[1, 1], [6, 1], [3, 5]]},
        ],
    }
    annotation = source_json / "IMG000001.json"
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    unmatched_image = source_images / "IMG000002.jpeg"
    Image.new("RGB", (8, 6), (90, 80, 70)).save(unmatched_image)
    unmatched_json = source_json / "IMG000002.json"
    unmatched_json.write_text(
        json.dumps({**payload, "imagePath": unmatched_image.name}), encoding="utf-8"
    )

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_exact",
                "image_id": main_image.name,
                "absolute_path": str(main_image),
                "sha256": _hash(source),
                "is_eligible": True,
                "assigned_split": "train",
                "subtype_label": "",
                "risk_label": "",
            }
        ]
    )
    manifest_path = manifest_dir / "dataset_clean.csv"
    manifest.to_csv(manifest_path, index=False)
    schema_path = annotation_root / "label_schema.yaml"
    schema_path.write_text(
        yaml.safe_dump({"subtype": {"classes": ["osteosarcoma"]}, "risk": {"classes": []}}),
        encoding="utf-8",
    )
    data_path = project / "configs" / "data.yaml"
    data_path.parent.mkdir()
    data_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(project),
                "dataset_root": str(project / "dataset" / "Main"),
                "source_read_only": True,
                "schema_path": str(schema_path),
                "provenance": {"btxrd_root": "dataset/BTXRD"},
            }
        ),
        encoding="utf-8",
    )
    config_path = project / "configs" / "annotation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_config": str(data_path),
                "annotation_root": str(annotation_root),
                "manifest": str(manifest_path),
                "database": str(annotation_root / "reviews.sqlite3"),
                "concept_names": [],
                "spatial_labels": ["lesion"],
            }
        ),
        encoding="utf-8",
    )
    return config_path, {
        "source": source,
        "annotation": annotation,
        "unmatched_image": unmatched_image,
        "unmatched_json": unmatched_json,
        "main": main_image,
        "manifest": manifest_path,
        "root": annotation_root,
    }


def _add_exact_source_duplicate(paths: dict[str, Path], *, conflicting: bool) -> Path:
    duplicate_image = paths["source"].with_name("IMG000003.jpeg")
    duplicate_image.write_bytes(paths["source"].read_bytes())
    payload = json.loads(paths["annotation"].read_text(encoding="utf-8"))
    payload["imagePath"] = duplicate_image.name
    if conflicting:
        payload["shapes"][0]["points"] = [[2, 1], [6, 4]]
    duplicate_json = paths["annotation"].with_name("IMG000003.json")
    duplicate_json.write_text(json.dumps(payload), encoding="utf-8")
    return duplicate_json


def test_import_exports_only_exact_hash_matches_and_preserves_source(tmp_path: Path) -> None:
    from bone_cancer_ai.annotations.source_import import import_source_annotations

    config, paths = _fixture(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items() if path.is_file()}

    report = import_source_annotations(config)

    assert report["source_json_count"] == 2
    assert report["converted_json_count"] == 1
    assert report["exported_sample_rows"] == 1
    assert report["invalid_json_count"] == 0
    assert report["unmatched_json_count"] == 1
    assert report["mask_rows"] == 1
    assert report["box_rows"] == 1
    master = pd.read_csv(report["master_csv"], keep_default_na=False)
    assert master[["sample_id", "image_id", "assigned_split"]].to_dict("records") == [
        {"sample_id": "sample_exact", "image_id": "opaque-main.jpeg", "assigned_split": "train"}
    ]
    row = master.iloc[0]
    assert row["source_sha256"] == _hash(paths["source"])
    assert row["source_json_sha256"] == _hash(paths["annotation"])
    assert row["status"] == "mechanically_validated"
    assert row["provenance"] == "source_dataset"
    assert row["reviewer"] == ""
    assert row["clinical_verification"] == "Unavailable"
    assert row["subtype_label"] == row["risk_label"] == ""
    assert Path(row["mask_path"]).is_file()
    assert Path(row["box_path"]).is_file()
    assert Path(report["generation_csv"]).is_file()
    issues = pd.read_csv(report["issues_csv"], keep_default_na=False)
    assert issues[["source_json_path", "error_code"]].to_dict("records") == [
        {
            "source_json_path": str(paths["unmatched_json"].resolve()),
            "error_code": "no_eligible_exact_hash_match",
        }
    ]
    assert {name: path.read_bytes() for name, path in paths.items() if path.is_file()} == before


def test_import_rejects_invalid_class_without_partial_artifacts(tmp_path: Path) -> None:
    from bone_cancer_ai.annotations.source_import import import_source_annotations

    config, paths = _fixture(tmp_path, label="diagnosis-from-filename")

    report = import_source_annotations(config)

    assert report["converted_json_count"] == 0
    assert report["exported_sample_rows"] == 0
    assert report["invalid_json_count"] == 1
    issues = pd.read_csv(report["issues_csv"], keep_default_na=False)
    invalid = issues.loc[issues.error_code.eq("invalid_labelme")].iloc[0]
    assert "Invalid class label" in invalid.error
    assert not list(Path(report["generation_root"]).glob("masks/*.png"))


def test_import_rechecks_main_bytes_before_transfer(tmp_path: Path) -> None:
    from bone_cancer_ai.annotations.source_import import import_source_annotations

    config, paths = _fixture(tmp_path)
    Image.new("RGB", (8, 6), (1, 2, 3)).save(paths["main"])

    report = import_source_annotations(config)

    assert report["exported_sample_rows"] == 0
    issues = pd.read_csv(report["issues_csv"], keep_default_na=False)
    assert "eligible_main_hash_mismatch" in set(issues.error_code)


def test_validate_source_import_detects_mask_drift(tmp_path: Path) -> None:
    from bone_cancer_ai.annotations.source_import import (
        import_source_annotations,
        validate_source_import,
    )

    config, _ = _fixture(tmp_path)
    report = import_source_annotations(config)
    assert validate_source_import(config) == {
        "valid": True,
        "checked_rows": 1,
        "issue_count": 0,
        "issues": [],
        "master_csv": report["master_csv"],
    }
    row = pd.read_csv(report["master_csv"]).iloc[0]
    Image.new("L", (8, 6), 0).save(row.mask_path)

    validation = validate_source_import(config)

    assert validation["valid"] is False
    assert validation["issue_count"] == 1
    assert validation["issues"][0]["error_code"] == "mask_content_mismatch"


def test_validate_source_import_rejects_artifact_path_outside_generation(
    tmp_path: Path,
) -> None:
    from bone_cancer_ai.annotations.source_import import (
        import_source_annotations,
        validate_source_import,
    )

    config, paths = _fixture(tmp_path)
    report = import_source_annotations(config)
    master = pd.read_csv(report["master_csv"], keep_default_na=False)
    master.loc[0, "box_path"] = str(paths["annotation"])
    master.to_csv(report["master_csv"], index=False)

    validation = validate_source_import(config)

    assert validation["valid"] is False
    assert "artifact_path_outside_generation" in {
        item["error_code"] for item in validation["issues"]
    }


def test_import_requires_safe_json_image_basename(tmp_path: Path) -> None:
    from bone_cancer_ai.annotations.source_import import import_source_annotations

    config, paths = _fixture(tmp_path)
    payload = json.loads(paths["annotation"].read_text(encoding="utf-8"))
    payload["imagePath"] = "../IMG000001.jpeg"
    paths["annotation"].write_text(json.dumps(payload), encoding="utf-8")

    report = import_source_annotations(config)

    assert report["invalid_json_count"] == 1
    issues = pd.read_csv(report["issues_csv"], keep_default_na=False)
    assert "unsafe_image_basename" in set(issues.error_code)


def test_import_quarantines_all_conflicting_annotations_for_exact_source_image(
    tmp_path: Path,
) -> None:
    from bone_cancer_ai.annotations.source_import import (
        import_source_annotations,
        validate_source_import,
    )

    config, paths = _fixture(tmp_path)
    duplicate_json = _add_exact_source_duplicate(paths, conflicting=True)

    report = import_source_annotations(config)

    assert report["converted_json_count"] == 0
    assert report["exported_sample_rows"] == 0
    assert report["invalid_json_count"] == 2
    issues = pd.read_csv(report["issues_csv"], keep_default_na=False)
    conflict_rows = issues.loc[issues.error_code.eq("conflicting_source_annotations")]
    assert set(conflict_rows.source_json_path) == {
        str(paths["annotation"].resolve()),
        str(duplicate_json.resolve()),
    }
    generation = Path(report["generation_root"])
    assert not list(generation.glob("masks/*.png"))
    assert not list(generation.glob("boxes/*.json"))
    assert validate_source_import(config)["valid"] is True


def test_import_deduplicates_equivalent_annotations_for_exact_source_image(
    tmp_path: Path,
) -> None:
    from bone_cancer_ai.annotations.source_import import import_source_annotations

    config, paths = _fixture(tmp_path)
    duplicate_json = _add_exact_source_duplicate(paths, conflicting=False)

    report = import_source_annotations(config)

    assert report["converted_json_count"] == 2
    assert report["exported_sample_rows"] == 1
    master = pd.read_csv(report["master_csv"], keep_default_na=False)
    assert master.sample_id.tolist() == ["sample_exact"]
    assert master.source_json_path.tolist() == [str(paths["annotation"].resolve())]
    assert json.loads(master.source_json_aliases.iloc[0]) == [str(duplicate_json.resolve())]


def test_validate_source_import_rejects_duplicate_rows_and_latest_generation_drift(
    tmp_path: Path,
) -> None:
    from bone_cancer_ai.annotations.source_import import (
        import_source_annotations,
        validate_source_import,
    )

    config, _ = _fixture(tmp_path)
    report = import_source_annotations(config)
    master = pd.read_csv(report["master_csv"], keep_default_na=False)
    pd.concat([master, master], ignore_index=True).to_csv(report["master_csv"], index=False)

    validation = validate_source_import(config)

    assert validation["valid"] is False
    assert {item["error_code"] for item in validation["issues"]} >= {
        "duplicate_sample_id",
        "latest_generation_index_mismatch",
    }
