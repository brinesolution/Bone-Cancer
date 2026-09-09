"""Duplicate tests use hand-checked 64-bit hashes and deterministic record IDs."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def row(record, sha, phash, canonical, label, width=100, height=100):
    split, image_id = record.split("/", 1)
    return {
        "record_id": record,
        "image_id": image_id,
        "official_split": split,
        "sha256": sha,
        "phash": phash,
        "canonical_source_id": canonical,
        "binary_label": label,
        "width": width,
        "height": height,
    }


def test_hamming_distance_uses_all_64_bits():
    from bone_cancer_ai.data.dedupe import hamming_distance

    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64
    with pytest.raises(ValueError):
        hamming_distance("short", "0000000000000000")


def test_duplicate_edges_merge_evidence_and_flag_leakage():
    from bone_cancer_ai.data.dedupe import find_duplicate_edges

    master = pd.DataFrame(
        [
            row("train/a.jpg", "same", "0000000000000000", "a", 0),
            row("val/a-copy.jpg", "same", "0000000000000000", "a-copy", 1),
            row("train/b-1.jpg", "b1", "1000000000000000", "b", 1),
            row("test/b-2.jpg", "b2", "1000000000000001", "b", 1),
            row("train/c.jpg", "c", "f000000000000000", "c", 0, 100, 40),
            row("train/d.jpg", "d", "f000000000000000", "d", 0, 30, 120),
        ]
    )
    edges = find_duplicate_edges(master, max_phash_distance=1)
    exact = edges[
        (edges.record_id == "train/a.jpg") & (edges.other_record_id == "val/a-copy.jpg")
    ].iloc[0]
    assert exact.match_type == "exact_sha256|perceptual_phash"
    assert bool(exact.cross_split) is True and bool(exact.label_conflict) is True
    canonical = edges[
        (edges.record_id == "test/b-2.jpg") | (edges.other_record_id == "test/b-2.jpg")
    ]
    canonical = canonical[canonical.match_type.str.contains("canonical_source_id")].iloc[0]
    assert canonical.phash_distance == 1
    aspect = edges[
        (edges.record_id == "train/c.jpg") & (edges.other_record_id == "train/d.jpg")
    ].iloc[0]
    assert aspect.match_type == "perceptual_phash"
    assert (aspect.width, aspect.height, aspect.other_width, aspect.other_height) == (
        100,
        40,
        30,
        120,
    )


def test_connected_groups_and_summary_are_deterministic():
    from bone_cancer_ai.data.dedupe import (
        build_duplicate_groups,
        find_duplicate_edges,
        summarize_leakage,
    )

    master = pd.DataFrame(
        [
            row("train/a.jpg", "one", "0000000000000000", "a", 0),
            row("val/b.jpg", "one", "0000000000000001", "b", 0),
            row("test/c.jpg", "three", "0000000000000003", "b", 0),
            row("train/z.jpg", "z", "ffffffffffffffff", "z", 1),
        ]
    )
    grouped = build_duplicate_groups(master, find_duplicate_edges(master, 1))
    assert grouped.duplicate_group.unique().tolist() == ["dup_000001"]
    assert set(grouped.record_id) | set(grouped.other_record_id) == {
        "train/a.jpg",
        "val/b.jpg",
        "test/c.jpg",
    }
    summary = summarize_leakage(grouped)
    assert summary["duplicate_groups"] == 1
    assert summary["images_in_duplicate_groups"] == 3
    assert summary["cross_split_edges"] == 2
    assert summary["cross_split_groups"] == 1
    assert summary["cross_split_exact_edges"] == 1


def test_duplicate_search_handles_moderate_unique_manifest():
    from bone_cancer_ai.data.dedupe import find_duplicate_edges

    generator = random.Random(42)
    hashes = set()
    while len(hashes) < 512:
        hashes.add(f"{generator.getrandbits(64):016x}")
    master = pd.DataFrame(
        [
            row(f"train/{index}.jpg", f"sha-{index}", phash, f"source-{index}", index % 2)
            for index, phash in enumerate(sorted(hashes))
        ]
    )
    assert find_duplicate_edges(master, max_phash_distance=0).empty


def test_duplicate_edges_reject_nonunique_record_ids():
    from bone_cancer_ai.data.dedupe import find_duplicate_edges

    master = pd.DataFrame([row("train/a.jpg", "a", "0" * 16, "a", 0)] * 2)
    with pytest.raises(ValueError, match="record_id"):
        find_duplicate_edges(master)


def test_duplicate_cli_writes_pair_report_and_summary(tmp_path):
    project = tmp_path / "project"
    manifest = project / "workspace/manifests/dataset_master.csv"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            row("train/a.jpg", "same", "0" * 16, "a", 0),
            row("test/b.jpg", "same", "0" * 16, "b", 0),
        ]
    ).to_csv(manifest, index=False)
    source = project / "dataset/Main"
    for split in ("train", "val", "test"):
        (source / split).mkdir(parents=True)
    config = project / "data.yaml"
    config.write_text(
        f'project_root: "{project.as_posix()}"\ndataset_root: "{source.as_posix()}"\n'
        "source_read_only: true\nofficial_splits: [train, val, test]\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts/03_find_duplicates.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = pd.read_csv(project / "workspace/manifests/duplicate_groups.csv")
    assert report["cross_split"].tolist() == [True]
    summary = json.loads((project / "logs/duplicate_summary.json").read_text())
    assert summary["duplicate_groups"] == 1
