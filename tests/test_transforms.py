"""Phase 3 preprocessing, transform, dataset, and DataLoader tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_decode_radiograph_supports_source_modes(tmp_path, mode):
    from bone_cancer_ai.data.preprocessing import decode_radiograph

    if mode == "L":
        array = np.arange(60, dtype=np.uint8).reshape(6, 10)
    elif mode == "RGB":
        array = np.zeros((6, 10, 3), dtype=np.uint8)
        array[..., 0], array[..., 1], array[..., 2] = 20, 80, 140
    else:
        array = np.zeros((6, 10, 4), dtype=np.uint8)
        array[..., :3], array[..., 3] = (20, 80, 140), 255
    path = tmp_path / f"source-{mode}.png"
    Image.fromarray(array, mode=mode).save(path)
    grayscale = decode_radiograph(path, force_grayscale=True)
    assert grayscale.shape == (6, 10, 3) and grayscale.dtype == np.uint8
    assert np.array_equal(grayscale[..., 0], grayscale[..., 1])
    assert np.array_equal(grayscale[..., 1], grayscale[..., 2])
    color = decode_radiograph(path, force_grayscale=False)
    assert color.shape == (6, 10, 3)


def test_resize_and_pad_preserves_aspect_ratio_and_mask_geometry():
    from bone_cancer_ai.data.preprocessing import resize_and_pad

    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[2:8, 5:15] = 255
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 5:15] = 1
    resized, resized_mask, metadata = resize_and_pad(image, mask, size=40, pad_value=0)
    assert resized.shape == (40, 40, 3)
    assert resized_mask.shape == (40, 40)
    assert metadata == {
        "original_height": 10,
        "original_width": 20,
        "resized_height": 20,
        "resized_width": 40,
        "pad_top": 10,
        "pad_bottom": 10,
        "pad_left": 0,
        "pad_right": 0,
    }
    assert resized[:10].max() == 0 and resized[30:].max() == 0
    assert set(np.unique(resized_mask)) <= {0, 1}


def test_validation_transform_is_deterministic_and_has_expected_shape():
    from bone_cancer_ai.data.transforms import RadiographTransform

    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    transform = RadiographTransform(image_size=64, train=False, seed=42)
    first = transform(image=image)
    second = transform(image=image)
    assert first["image"].shape == (3, 64, 64)
    assert first["mask"].shape == (1, 64, 64)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["mask"], second["mask"])
    assert torch.isfinite(first["image"]).all()


def test_seeded_training_transform_reproduces_sequence_and_forbids_flips():
    from bone_cancer_ai.data.transforms import RadiographTransform

    image = np.tile(np.arange(80, dtype=np.uint8), (60, 1))[..., None]
    image = np.repeat(image, 3, axis=2)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[15:45, 20:60] = 1
    augmentation = {
        "rotation_limit": 7,
        "translate_limit": 0.03,
        "scale_limit": 0.05,
        "brightness_contrast_limit": 0.1,
        "gamma_limit": [90, 110],
        "gaussian_noise_std_range": [0.01, 0.03],
    }
    first = RadiographTransform(64, train=True, seed=7, augmentation=augmentation)
    second = RadiographTransform(64, train=True, seed=7, augmentation=augmentation)
    first_sequence = [first(image=image, mask=mask) for _ in range(3)]
    second_sequence = [second(image=image, mask=mask) for _ in range(3)]
    for left, right in zip(first_sequence, second_sequence, strict=True):
        assert torch.equal(left["image"], right["image"])
        assert torch.equal(left["mask"], right["mask"])
    names = first.augmentation_names
    assert "HorizontalFlip" not in names and "VerticalFlip" not in names
    assert not any("Crop" in name for name in names)


def test_image_and_mask_remain_aligned_after_validation_geometry():
    from bone_cancer_ai.data.transforms import RadiographTransform

    image = np.zeros((30, 60, 3), dtype=np.uint8)
    image[8:22, 20:40] = 255
    mask = np.zeros((30, 60), dtype=np.uint8)
    mask[8:22, 20:40] = 1
    transformed = RadiographTransform(96, train=False, seed=2, normalize=False)(
        image=image, mask=mask
    )
    image_support = transformed["image"][0] > 0.5
    mask_support = transformed["mask"][0].bool()
    intersection = (image_support & mask_support).sum()
    union = (image_support | mask_support).sum()
    assert float(intersection / union) > 0.95


def _schema(*, optional: bool = False, masks: bool = False) -> dict:
    return {
        "binary": {"enabled": True, "classes": {0: "normal", 1: "bone_tumour"}},
        "subtype": {
            "enabled": optional,
            "classes": ["osteochondroma", "osteosarcoma"],
        },
        "risk": {"enabled": optional, "classes": ["benign", "malignant"]},
        "concepts": {"enabled": False, "names": []},
        "localization": {"masks_verified": masks},
    }


def _dataset_row(path, *, sample_id="sample_1", split="fit_train", label=1) -> dict:
    return {
        "sample_id": sample_id,
        "record_id": f"train/{path.name}",
        "image_id": path.name,
        "absolute_path": str(path),
        "official_split": "train",
        "assigned_split": split,
        "split_generation_id": "fixture_generation",
        "split_group_id": f"group_{sample_id}",
        "binary_label": label,
        "subtype_label": "osteosarcoma",
        "risk_label": "malignant",
        "has_concepts": False,
        "has_mask": False,
        "width": 20,
        "height": 12,
        "is_eligible": True,
    }


def test_dataset_returns_partial_task_masks_and_opaque_metadata(tmp_path):
    from bone_cancer_ai.data.dataset import BoneCancerDataset
    from bone_cancer_ai.data.transforms import RadiographTransform

    path = tmp_path / "femur_osteosarcoma_case.png"
    Image.fromarray(np.arange(240, dtype=np.uint8).reshape(12, 20)).save(path)
    dataset = BoneCancerDataset(
        pd.DataFrame([_dataset_row(path)]),
        RadiographTransform(64, train=False, seed=42),
        _schema(optional=False),
        force_grayscale=True,
    )
    item = dataset[0]
    assert set(item) == {
        "image",
        "binary_label",
        "subtype_label",
        "risk_label",
        "concept_labels",
        "mask",
        "valid_task_mask",
        "metadata",
    }
    assert item["image"].shape == (3, 64, 64) and item["mask"].shape == (1, 64, 64)
    assert item["binary_label"].item() == 1
    assert item["subtype_label"].item() == -1 and item["risk_label"].item() == -1
    assert item["valid_task_mask"]["binary"].item() is True
    assert item["valid_task_mask"]["subtype"].item() is False
    assert item["valid_task_mask"]["risk"].item() is False
    assert item["valid_task_mask"]["mask"].item() is False
    assert item["valid_task_mask"]["sample"].item() is True
    assert item["metadata"]["sample_id"] == "sample_1"
    assert "femur" not in repr(item) and str(tmp_path) not in repr(item)


def test_dataset_encodes_enabled_optional_labels_and_aligned_mask(tmp_path):
    from bone_cancer_ai.data.dataset import BoneCancerDataset
    from bone_cancer_ai.data.transforms import RadiographTransform

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    image = np.zeros((12, 20), dtype=np.uint8)
    image[3:9, 5:15] = 255
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[3:9, 5:15] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    row = _dataset_row(image_path)
    row.update({"mask_path": str(mask_path), "has_mask": True})
    item = BoneCancerDataset(
        pd.DataFrame([row]),
        RadiographTransform(64, train=False, seed=1, normalize=False),
        _schema(optional=True, masks=True),
    )[0]
    assert item["subtype_label"].item() == 1 and item["risk_label"].item() == 1
    assert item["valid_task_mask"]["subtype"].item() is True
    assert item["valid_task_mask"]["risk"].item() is True
    assert item["valid_task_mask"]["mask"].item() is True
    support = item["image"][0] > 0.5
    mask_support = item["mask"][0].bool()
    assert float((support & mask_support).sum() / (support | mask_support).sum()) > 0.9


def test_dataset_treats_malformed_concepts_as_missing_partial_labels(tmp_path):
    from bone_cancer_ai.data.dataset import BoneCancerDataset
    from bone_cancer_ai.data.transforms import RadiographTransform

    image_path = tmp_path / "image.png"
    Image.fromarray(np.arange(240, dtype=np.uint8).reshape(12, 20)).save(image_path)
    row = _dataset_row(image_path)
    row["concept_labels"] = "{not-json"
    schema = _schema()
    schema["concepts"] = {"enabled": True, "names": ["cortical_disruption"]}
    item = BoneCancerDataset(
        pd.DataFrame([row]),
        RadiographTransform(32, train=False, seed=1),
        schema,
    )[0]
    assert item["concept_labels"].tolist() == [-1.0]
    assert item["valid_task_mask"]["concepts"].tolist() == [False]


def test_dataset_pixels_do_not_depend_on_diagnostic_filename(tmp_path):
    from bone_cancer_ai.data.dataset import BoneCancerDataset
    from bone_cancer_ai.data.transforms import RadiographTransform

    pixels = np.arange(240, dtype=np.uint8).reshape(12, 20)
    first_path = tmp_path / "osteosarcoma_secret.png"
    second_path = tmp_path / "normal_secret.png"
    Image.fromarray(pixels).save(first_path)
    Image.fromarray(pixels).save(second_path)
    frame = pd.DataFrame(
        [
            _dataset_row(first_path, sample_id="sample_a", label=0),
            _dataset_row(second_path, sample_id="sample_b", label=0),
        ]
    )
    dataset = BoneCancerDataset(
        frame,
        RadiographTransform(64, train=False, seed=42),
        _schema(),
    )
    first, second = dataset[0], dataset[1]
    assert torch.equal(first["image"], second["image"])
    assert first["binary_label"].item() == second["binary_label"].item() == 0
    assert "osteosarcoma" not in repr(first) and "normal_secret" not in repr(second)


def test_dataset_contains_runtime_decode_failure_without_relabeling(tmp_path):
    from bone_cancer_ai.data.dataset import BoneCancerDataset
    from bone_cancer_ai.data.transforms import RadiographTransform

    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not an image")
    item = BoneCancerDataset(
        pd.DataFrame([_dataset_row(path)]),
        RadiographTransform(32, train=False, seed=42),
        _schema(),
    )[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["image"].count_nonzero() == 0
    assert item["binary_label"].item() == 1
    assert not any(bool(value.any()) for value in item["valid_task_mask"].values())
    assert item["metadata"]["load_error"] == "UnidentifiedImageError"
    assert "corrupt.jpg" not in repr(item)


def test_dataloader_completes_mixed_valid_and_corrupt_epoch(tmp_path):
    from torch.utils.data import DataLoader

    from bone_cancer_ai.data.dataset import BoneCancerDataset
    from bone_cancer_ai.data.transforms import RadiographTransform

    valid_path = tmp_path / "valid.png"
    corrupt_path = tmp_path / "corrupt.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(valid_path)
    corrupt_path.write_bytes(b"invalid pixels")
    frame = pd.DataFrame(
        [
            _dataset_row(corrupt_path, sample_id="sample_a"),
            _dataset_row(valid_path, sample_id="sample_b"),
        ]
    )
    dataset = BoneCancerDataset(
        frame,
        RadiographTransform(32, train=False, seed=42),
        _schema(),
    )
    batches = list(DataLoader(dataset, batch_size=2, shuffle=False))
    assert len(batches) == 1
    assert batches[0]["image"].shape == (2, 3, 32, 32)
    assert batches[0]["valid_task_mask"]["sample"].tolist() == [False, True]


def test_datamodule_loads_deterministic_batches_from_every_split(tmp_path):
    from bone_cancer_ai.data.datamodule import BoneCancerDataModule, smoke_test_loaders

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split_index, split in enumerate(("fit_train", "calibration", "val", "test")):
        rows = []
        for index in range(3):
            path = tmp_path / f"source-{split_index}-{index}.png"
            Image.fromarray(
                np.full((12 + index, 20), 20 + split_index * 30 + index, dtype=np.uint8)
            ).save(path)
            rows.append(
                _dataset_row(
                    path,
                    sample_id=f"sample_{split}_{index}",
                    split=split,
                    label=index % 2,
                )
            )
        pd.DataFrame(rows).to_csv(split_dir / f"{split}.csv", index=False)
    _stamp_split_generation(split_dir)
    config = {
        "dataset_root": str(tmp_path),
        "image_size": 48,
        "seed": 13,
        "preprocessing": {"force_grayscale": True, "pad_value": 0},
        "augmentation": {"rotation_limit": 0, "translate_limit": 0, "scale_limit": 0},
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "sampling": "shuffle",
        },
    }
    first = BoneCancerDataModule(split_dir, config, _schema())
    second = BoneCancerDataModule(split_dir, config, _schema())
    first.setup()
    second.setup()
    first_loaders = first.loaders()
    second_loaders = second.loaders()
    assert set(first_loaders) == {"fit_train", "calibration", "val", "test"}
    for split, loader in first_loaders.items():
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 3, 48, 48)
        assert batch["mask"].shape == (2, 1, 48, 48)
        assert batch["metadata"]["assigned_split"] == [split, split]
        assert "absolute_path" not in batch and "image_id" not in batch
    first_val = next(iter(first_loaders["val"]))
    second_val = next(iter(second_loaders["val"]))
    assert torch.equal(first_val["image"], second_val["image"])
    figure = tmp_path / "batch-grid.png"
    smoke = smoke_test_loaders(first.loaders(), figure, image_size=48)
    assert set(smoke) == {"fit_train", "calibration", "val", "test"}
    assert all(value["image_shape"] == [2, 3, 48, 48] for value in smoke.values())
    with Image.open(figure) as rendered:
        assert rendered.format == "PNG" and rendered.width > 400


def test_datamodule_can_use_weighted_training_sampler(tmp_path):
    from torch.utils.data import WeightedRandomSampler

    from bone_cancer_ai.data.datamodule import BoneCancerDataModule

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    image_path = tmp_path / "image.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    for split in ("fit_train", "calibration", "val", "test"):
        rows = []
        for index, label in enumerate([0, 0, 0, 1]):
            row = _dataset_row(
                image_path, sample_id=f"{split}_{index}", split=split, label=label
            )
            row["record_id"] = f"{split}/image-{index}.png"
            row["official_split"] = "train" if split in {"fit_train", "calibration"} else split
            rows.append(row)
        pd.DataFrame(rows).to_csv(split_dir / f"{split}.csv", index=False)
    _stamp_split_generation(split_dir)
    config = {
        "dataset_root": str(tmp_path),
        "image_size": 32,
        "seed": 7,
        "preprocessing": {"force_grayscale": True},
        "augmentation": {},
        "loader": {"batch_size": 2, "num_workers": 0, "sampling": "weighted"},
    }
    module = BoneCancerDataModule(split_dir, config, _schema())
    module.setup()
    assert isinstance(module.loader("fit_train").sampler, WeightedRandomSampler)


def _stamp_split_generation(split_dir):
    from bone_cancer_ai.data.samplers import split_generation_id

    names = ("fit_train", "calibration", "val", "test")
    frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in names}
    generation = split_generation_id(pd.concat(frames.values(), ignore_index=True))
    for name, frame in frames.items():
        frame["split_generation_id"] = generation
        frame.to_csv(split_dir / f"{name}.csv", index=False)


def _write_datamodule_split_fixtures(split_dir, image_path):
    split_dir.mkdir()
    for split in ("fit_train", "calibration", "val", "test"):
        row = _dataset_row(image_path, sample_id=f"sample_{split}", split=split)
        row["record_id"] = f"{split}/image.png"
        row["official_split"] = "train" if split in {"fit_train", "calibration"} else split
        row["split_group_id"] = f"group_{split}"
        pd.DataFrame([row]).to_csv(split_dir / f"{split}.csv", index=False)
    _stamp_split_generation(split_dir)


def test_datamodule_rejects_cross_split_group_and_stale_generation(tmp_path):
    from bone_cancer_ai.data.datamodule import BoneCancerDataModule

    source = tmp_path / "dataset/Main"
    source.mkdir(parents=True)
    image_path = source / "image.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    split_dir = tmp_path / "splits"
    _write_datamodule_split_fixtures(split_dir, image_path)
    val = pd.read_csv(split_dir / "val.csv")
    val["split_group_id"] = "group_fit_train"
    val["split_generation_id"] = "stale_generation"
    val.to_csv(split_dir / "val.csv", index=False)
    module = BoneCancerDataModule(split_dir, {"dataset_root": str(source)}, _schema())
    with pytest.raises(ValueError, match="generation|group overlap"):
        module.setup()


def test_datamodule_rejects_manifest_path_outside_dataset_root(tmp_path):
    from bone_cancer_ai.data.datamodule import BoneCancerDataModule

    source = tmp_path / "dataset/Main"
    source.mkdir(parents=True)
    image_path = source / "image.png"
    outside_path = tmp_path / "outside.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(outside_path)
    split_dir = tmp_path / "splits"
    _write_datamodule_split_fixtures(split_dir, image_path)
    test = pd.read_csv(split_dir / "test.csv")
    test["absolute_path"] = str(outside_path)
    test.to_csv(split_dir / "test.csv", index=False)
    _stamp_split_generation(split_dir)
    module = BoneCancerDataModule(split_dir, {"dataset_root": str(source)}, _schema())
    with pytest.raises(ValueError, match="outside dataset_root"):
        module.setup()


def test_datamodule_rejects_stale_label_with_old_generation(tmp_path):
    from bone_cancer_ai.data.datamodule import BoneCancerDataModule

    source = tmp_path / "dataset/Main"
    source.mkdir(parents=True)
    image_path = source / "image.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    split_dir = tmp_path / "splits"
    _write_datamodule_split_fixtures(split_dir, image_path)
    calibration = pd.read_csv(split_dir / "calibration.csv")
    calibration["binary_label"] = 0
    calibration.to_csv(split_dir / "calibration.csv", index=False)
    module = BoneCancerDataModule(split_dir, {"dataset_root": str(source)}, _schema())
    with pytest.raises(ValueError, match="generation fingerprint"):
        module.setup()


def test_optional_cache_cli_is_deterministic_and_keeps_source_unchanged(tmp_path):
    from bone_cancer_ai.data.schema import opaque_id

    project = tmp_path / "project"
    source = project / "dataset/Main/train"
    source.mkdir(parents=True)
    image_path = source / "source.png"
    Image.fromarray(np.arange(240, dtype=np.uint8).reshape(12, 20)).save(image_path)
    source_before = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_dir = project / "workspace/manifests"
    manifest_dir.mkdir(parents=True)
    row = _dataset_row(image_path, sample_id=opaque_id("sample", "train/source.png"))
    row.update({"sha256": source_before, "clean_pool": "train_pool"})
    pd.DataFrame([row]).to_csv(manifest_dir / "dataset_clean.csv", index=False)
    config = project / "data.yaml"
    config.write_text(
        "\n".join(
            [
                f'project_root: "{project.as_posix()}"',
                f'dataset_root: "{(project / "dataset/Main").as_posix()}"',
                "source_read_only: true",
                "image_size: 32",
                "preprocessing:",
                "  force_grayscale: true",
                "  pad_value: 0",
            ]
        ),
        encoding="utf-8",
    )
    repository = tmp_path.parents[2]
    script = repository / "scripts/05_build_cache.py"
    command = [sys.executable, str(script), "--config", str(config), "--limit", "1"]
    first = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr + first.stdout
    cache_path = next((project / "workspace/cache/resized").glob("sample_*.png"))
    first_cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    second = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr + second.stdout
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == first_cache_hash
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == source_before
    index = pd.read_csv(project / "workspace/cache/resized/cache_index.csv")
    assert index["source_sha256"].tolist() == [source_before]
    summary = json.loads((project / "logs/cache_summary.json").read_text())
    assert summary["cached"] == 1 and summary["failed"] == 0


def test_cache_removes_stale_output_when_source_hash_changes(tmp_path):
    from bone_cancer_ai.data.preprocessing import cache_manifest_images

    source = tmp_path / "source.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_safe",
                "absolute_path": str(source),
                "sha256": source_hash,
                "is_eligible": True,
            }
        ]
    )
    cache_dir = tmp_path / "cache"
    first, _ = cache_manifest_images(
        manifest,
        cache_dir,
        image_size=32,
        force_grayscale=True,
        pad_value=0,
    )
    cache_path = Path(first.iloc[0]["cache_path"])
    assert cache_path.is_file()
    source.write_bytes(b"changed after manifest audit")
    second, summary = cache_manifest_images(
        manifest,
        cache_dir,
        image_size=32,
        force_grayscale=True,
        pad_value=0,
    )
    assert summary["failed"] == 1
    assert second.iloc[0]["error_type"] == "SourceHashMismatch"
    assert not cache_path.exists()
