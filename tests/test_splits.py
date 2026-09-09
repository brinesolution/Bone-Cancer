"""Phase 3 clean-manifest and grouped-split tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


def _master_row(
    record_id: str,
    split: str,
    label: int | None,
    *,
    quality: str = "readable",
    exclude_reason: str = "",
) -> dict:
    image_id = record_id.split("/", 1)[1]
    return {
        "record_id": record_id,
        "image_id": image_id,
        "absolute_path": f"C:/immutable/{split}/{image_id}",
        "official_split": split,
        "sha256": f"sha-{record_id}",
        "binary_label": label,
        "binary_semantic": "normal" if label == 0 else "bone_tumour" if label == 1 else "",
        "subtype_label": "",
        "risk_label": "",
        "has_box": False,
        "has_mask": False,
        "has_concepts": False,
        "width": 100,
        "height": 80,
        "quality_status": quality,
        "exclude_reason": exclude_reason,
    }


def _edge(
    group: str,
    left: str,
    right: str,
    *,
    match_type: str = "canonical_source_id",
    label_conflict: bool = False,
) -> dict:
    return {
        "duplicate_group": group,
        "record_id": left,
        "other_record_id": right,
        "match_type": match_type,
        "label_conflict": label_conflict,
    }


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.DataFrame(
        [
            _master_row("train/a.jpg", "train", 0),
            _master_row("test/a-copy.jpg", "test", 0),
            _master_row("train/b.jpg", "train", 1),
            _master_row("val/b-copy.jpg", "val", 1),
            _master_row("train/conflict-a.jpg", "train", 0),
            _master_row("train/conflict-b.jpg", "train", 1),
            _master_row("train/near.jpg", "train", 0, quality="near_blank"),
            _master_row("train/broken.jpg", "train", 1, quality="unreadable"),
            _master_row("val/unique.jpg", "val", 0),
            _master_row("train/visual-a.jpg", "train", 0),
            _master_row("train/visual-b.jpg", "train", 1),
        ]
    )
    duplicates = pd.DataFrame(
        [
            _edge("dup_1", "train/a.jpg", "test/a-copy.jpg"),
            _edge("dup_2", "train/b.jpg", "val/b-copy.jpg"),
            _edge(
                "dup_3",
                "train/conflict-a.jpg",
                "train/conflict-b.jpg",
                match_type="exact_sha256|perceptual_phash",
                label_conflict=True,
            ),
            _edge(
                "dup_4",
                "train/visual-a.jpg",
                "train/visual-b.jpg",
                match_type="perceptual_phash",
                label_conflict=True,
            ),
        ]
    )
    return master, duplicates


def test_clean_manifest_preserves_heldout_and_logs_lower_priority_copies():
    from bone_cancer_ai.data.quality import build_clean_manifest

    master, duplicates = _fixture_frames()
    clean, exclusions, summary = build_clean_manifest(master, duplicates, {})
    assignments = clean.set_index("record_id")["clean_pool"].to_dict()
    assert assignments["test/a-copy.jpg"] == "test"
    assert assignments["train/a.jpg"] == "excluded"
    assert assignments["val/b-copy.jpg"] == "val"
    assert assignments["train/b.jpg"] == "excluded"
    reasons = exclusions.set_index("record_id")["exclude_reason"].to_dict()
    assert reasons["train/a.jpg"] == "cross_split_duplicate_lower_priority"
    assert reasons["train/b.jpg"] == "cross_split_duplicate_lower_priority"
    assert summary["official_records_reassigned_to_training"] == 0


def test_clean_manifest_excludes_exact_conflicts_and_unreadable_not_near_blank():
    from bone_cancer_ai.data.quality import build_clean_manifest

    master, duplicates = _fixture_frames()
    clean, exclusions, _ = build_clean_manifest(master, duplicates, {})
    indexed = clean.set_index("record_id")
    assert indexed.loc["train/conflict-a.jpg", "exclude_reason"] == (
        "exact_duplicate_label_conflict"
    )
    assert indexed.loc["train/conflict-b.jpg", "clean_pool"] == "excluded"
    assert indexed.loc["train/broken.jpg", "exclude_reason"] == "quality_unreadable"
    assert indexed.loc["train/near.jpg", "clean_pool"] == "train_pool"
    assert bool(indexed.loc["train/near.jpg", "review_required"]) is True
    assert "quality_near_blank" in indexed.loc["train/near.jpg", "review_reason"]
    assert set(exclusions["record_id"]) == {
        "train/a.jpg",
        "train/b.jpg",
        "train/conflict-a.jpg",
        "train/conflict-b.jpg",
        "train/broken.jpg",
    }


def test_clean_manifest_keeps_perceptual_conflict_group_together_for_review():
    from bone_cancer_ai.data.quality import build_clean_manifest

    master, duplicates = _fixture_frames()
    clean, _, _ = build_clean_manifest(master, duplicates, {})
    visual = clean[clean["duplicate_group"].eq("dup_4")]
    assert visual["clean_pool"].unique().tolist() == ["train_pool"]
    assert visual["review_required"].all()
    assert visual["review_reason"].str.contains("perceptual_label_conflict").all()
    singleton = clean.loc[clean.record_id.eq("val/unique.jpg")].iloc[0]
    assert singleton.split_group_id.startswith("single_")
    assert "unique.jpg" not in singleton.split_group_id
    assert singleton.sample_id.startswith("sample_")
    assert "unique.jpg" not in singleton.sample_id


def test_clean_manifest_is_deterministic_and_complete():
    from bone_cancer_ai.data.quality import build_clean_manifest, validate_clean_manifest

    master, duplicates = _fixture_frames()
    first = build_clean_manifest(master.sample(frac=1, random_state=3), duplicates, {})[0]
    second = build_clean_manifest(master, duplicates.sample(frac=1, random_state=4), {})[0]
    pd.testing.assert_frame_equal(first, second)
    validation = validate_clean_manifest(first, expected_records=set(master.record_id))
    assert validation == {
        "records": len(master),
        "eligible_records": 6,
        "excluded_records": 5,
        "record_ids_unique": True,
        "complete_coverage": True,
        "group_pool_overlap": 0,
    }


def test_clean_manifest_rejects_unknown_conflict_actions():
    from bone_cancer_ai.data.quality import build_clean_manifest

    master, duplicates = _fixture_frames()
    with pytest.raises(ValueError, match="exact_label_conflict_action"):
        build_clean_manifest(
            master,
            duplicates,
            {"exact_label_conflict_action": "silently_relabel"},
        )


def test_clean_manifest_cli_writes_safe_outputs(tmp_path):
    project = tmp_path / "project"
    source = project / "dataset/Main"
    source.mkdir(parents=True)
    manifest_dir = project / "workspace/manifests"
    manifest_dir.mkdir(parents=True)
    master, duplicates = _fixture_frames()
    master.to_csv(manifest_dir / "dataset_master.csv", index=False)
    duplicates.to_csv(manifest_dir / "duplicate_groups.csv", index=False)
    config = project / "data.yaml"
    config.write_text(
        "\n".join(
            [
                f'project_root: "{project.as_posix()}"',
                f'dataset_root: "{source.as_posix()}"',
                "source_read_only: true",
                "split_policy:",
                "  heldout_priority: [test, val, train]",
                "  excluded_quality_statuses: [unreadable, blank]",
            ]
        ),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/04_build_clean_manifest.py"),
            "--config",
            str(config),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    clean = pd.read_csv(manifest_dir / "dataset_clean.csv")
    exclusions = pd.read_csv(manifest_dir / "exclusions.csv")
    summary = json.loads((project / "logs/clean_manifest_summary.json").read_text())
    assert len(clean) == len(master) and len(exclusions) == 5
    assert summary["validation"]["group_pool_overlap"] == 0
    assert sorted(source.iterdir()) == []


def _train_pool_frame(groups: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(groups):
        group = f"group_{index:03d}"
        label = index % 2
        repeats = 2 if index % 7 == 0 else 1
        for repeat in range(repeats):
            record = f"train/item-{index}-{repeat}.jpg"
            row = _master_row(record, "train", label)
            row.update(
                {
                    "sample_id": f"sample_{index}_{repeat}",
                    "duplicate_group": group if repeats > 1 else "",
                    "split_group_id": group,
                    "clean_pool": "train_pool",
                    "assigned_split": "train_pool",
                    "is_eligible": True,
                    "review_required": False,
                    "review_reason": "",
                    "split_correction_reason": "",
                }
            )
            rows.append(row)
    rows.extend(
        [
            {
                **_master_row("val/heldout.jpg", "val", 0),
                "sample_id": "sample_val",
                "duplicate_group": "",
                "split_group_id": "heldout_val",
                "clean_pool": "val",
                "assigned_split": "val",
                "is_eligible": True,
                "review_required": False,
                "review_reason": "",
                "split_correction_reason": "",
            },
            {
                **_master_row("test/heldout.jpg", "test", 1),
                "sample_id": "sample_test",
                "duplicate_group": "",
                "split_group_id": "heldout_test",
                "clean_pool": "test",
                "assigned_split": "test",
                "is_eligible": True,
                "review_required": False,
                "review_reason": "",
                "split_correction_reason": "",
            },
        ]
    )
    return pd.DataFrame(rows)


def test_internal_split_is_deterministic_grouped_and_stratified():
    from bone_cancer_ai.data.samplers import assign_internal_splits, validate_final_splits

    clean = _train_pool_frame()
    first, first_summary = assign_internal_splits(
        clean.sample(frac=1, random_state=11),
        calibration_fraction=0.2,
        seed=42,
        search_trials=64,
    )
    second, _ = assign_internal_splits(
        clean.sample(frac=1, random_state=12),
        calibration_fraction=0.2,
        seed=42,
        search_trials=64,
    )
    pd.testing.assert_frame_equal(first, second)
    validation = validate_final_splits(first)
    assert validation["group_overlap"] == 0
    assert validation["record_overlap"] == 0
    assert validation["unassigned_eligible_records"] == 0
    train = first[first.clean_pool.eq("train_pool")]
    assert set(train.groupby("split_group_id")["assigned_split"].nunique()) == {1}
    assert set(train.groupby("assigned_split")["binary_label"].nunique()) == {2}
    assert abs(first_summary["actual_calibration_fraction"] - 0.2) < 0.08


def test_internal_split_does_not_use_heldout_labels_for_calibration_choice():
    from bone_cancer_ai.data.samplers import assign_internal_splits

    clean = _train_pool_frame()
    first, _ = assign_internal_splits(clean, 0.2, seed=9, search_trials=32)
    changed = clean.copy()
    changed.loc[changed.clean_pool.isin(["val", "test"]), "binary_label"] = [1, 0]
    second, _ = assign_internal_splits(changed, 0.2, seed=9, search_trials=32)
    first_train = first[first.clean_pool.eq("train_pool")][["record_id", "assigned_split"]]
    second_train = second[second.clean_pool.eq("train_pool")][["record_id", "assigned_split"]]
    pd.testing.assert_frame_equal(
        first_train.reset_index(drop=True), second_train.reset_index(drop=True)
    )


def test_internal_split_cli_writes_all_four_manifests(tmp_path):
    project = tmp_path / "project"
    source = project / "dataset/Main"
    source.mkdir(parents=True)
    manifest_dir = project / "workspace/manifests"
    manifest_dir.mkdir(parents=True)
    _train_pool_frame().to_csv(manifest_dir / "dataset_clean.csv", index=False)
    config = project / "data.yaml"
    config.write_text(
        "\n".join(
            [
                f'project_root: "{project.as_posix()}"',
                f'dataset_root: "{source.as_posix()}"',
                "source_read_only: true",
                "seed: 42",
                "split_policy:",
                "  calibration_fraction: 0.2",
                "  calibration_search_trials: 32",
            ]
        ),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/06_make_internal_splits.py"),
            "--config",
            str(config),
            "--skip-loader-smoke",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    split_dir = project / "workspace/splits"
    frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in (
        "fit_train", "calibration", "val", "test"
    )}
    assert all(not frame.empty for frame in frames.values())
    assert all(frame.assigned_split.eq(name).all() for name, frame in frames.items())
    summary = json.loads((project / "logs/internal_split_summary.json").read_text())
    assert summary["validation"]["group_overlap"] == 0


def test_binary_weights_and_weighted_sampler_are_deterministic():
    from bone_cancer_ai.data.samplers import binary_class_weights, make_weighted_sampler

    frame = pd.DataFrame({"binary_label": [0, 0, 0, 1]})
    assert torch.allclose(binary_class_weights(frame), torch.tensor([2 / 3, 2.0]))
    first = list(make_weighted_sampler(frame, seed=17))
    second = list(make_weighted_sampler(frame, seed=17))
    assert first == second and len(first) == len(frame)
