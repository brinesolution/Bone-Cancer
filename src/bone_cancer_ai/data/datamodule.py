"""Split-aware PyTorch DataLoaders for clean manifest partitions."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, get_worker_info

from bone_cancer_ai.data.dataset import BoneCancerDataset
from bone_cancer_ai.data.samplers import (
    make_weighted_sampler,
    split_generation_id,
    validate_final_splits,
)
from bone_cancer_ai.data.transforms import IMAGENET_MEAN, IMAGENET_STD, RadiographTransform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLIT_NAMES = ("fit_train", "calibration", "val", "test")


def seed_loader_worker(worker_id: int) -> None:
    """Give Python, NumPy and Albumentations independent deterministic worker streams."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker = get_worker_info()
    if worker is not None and hasattr(worker.dataset.transform, "set_seed"):
        worker.dataset.transform.set_seed(worker_seed)


class BoneCancerDataModule:
    """Create loaders from four self-contained, leakage-validated split manifests."""

    def __init__(
        self,
        split_dir: Path,
        data_config: dict[str, Any],
        label_schema: dict[str, Any],
        *,
        splits: tuple[str, ...] = SPLIT_NAMES,
    ) -> None:
        if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLIT_NAMES):
            raise ValueError("splits must be a nonempty unique subset of known split names")
        self.split_dir = Path(split_dir)
        self.config = data_config
        self.label_schema = label_schema
        self.splits = tuple(splits)
        dataset_root = self.config.get("dataset_root")
        if not dataset_root:
            raise ValueError("data_config must define dataset_root")
        self.dataset_root = Path(str(dataset_root)).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Missing dataset_root: {self.dataset_root}")
        self.frames: dict[str, pd.DataFrame] = {}
        self.datasets: dict[str, BoneCancerDataset] = {}

    def setup(self) -> None:
        image_size = int(self.config.get("image_size", 384))
        seed = int(self.config.get("seed", 42))
        preprocessing = self.config.get("preprocessing", {})
        augmentation = self.config.get("augmentation", {})
        frames: dict[str, pd.DataFrame] = {}
        for split in self.splits:
            path = self.split_dir / f"{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing split manifest: {path}")
            frame = pd.read_csv(path, keep_default_na=True)
            if frame.empty:
                raise ValueError(f"Split manifest is empty: {split}")
            if not frame["assigned_split"].eq(split).all():
                raise ValueError(f"Split manifest contains wrong assignments: {split}")
            frames[split] = frame
        combined = pd.concat(frames.values(), ignore_index=True)
        required = {
            "record_id",
            "sample_id",
            "split_group_id",
            "official_split",
            "assigned_split",
            "absolute_path",
            "split_generation_id",
            "is_eligible",
        }
        missing = required - set(combined.columns)
        if missing:
            raise ValueError(f"Split bundle missing columns: {sorted(missing)}")
        if not combined["is_eligible"].eq(True).all():
            raise ValueError("Split bundle contains excluded or invalid eligibility rows")
        validation = validate_final_splits(combined)
        if validation["record_overlap"]:
            raise ValueError("Split bundle has record overlap")
        if validation["sample_overlap"]:
            raise ValueError("Split bundle has sample overlap")
        if validation["group_overlap"]:
            raise ValueError("Split bundle has group overlap")
        if validation["official_heldout_records_assigned_to_training"]:
            raise ValueError("Split bundle moves official held-out records into training")
        if self.splits == SPLIT_NAMES and validation["generation_mismatch"]:
            raise ValueError("Split bundle generation fingerprint is stale or mixed")
        generation = combined["split_generation_id"].astype(str).iloc[0]
        if self.splits == SPLIT_NAMES and generation != split_generation_id(combined):
            raise ValueError("Split bundle generation fingerprint is stale or mixed")
        if combined["split_generation_id"].astype(str).nunique() != 1:
            raise ValueError("Split bundle generation fingerprint is stale or mixed")
        outside = [
            value
            for value in combined["absolute_path"]
            if not Path(str(value)).resolve().is_relative_to(self.dataset_root)
        ]
        if outside:
            raise ValueError("Split manifest contains a path outside dataset_root")

        for offset, split in enumerate(self.splits):
            frame = frames[split]
            transform = RadiographTransform(
                image_size,
                train=split == "fit_train",
                seed=seed + offset,
                augmentation=augmentation,
                pad_value=int(preprocessing.get("pad_value", 0)),
                normalize=preprocessing.get("normalize", "imagenet") == "imagenet",
            )
            self.frames[split] = frame
            self.datasets[split] = BoneCancerDataset(
                frame,
                transform,
                self.label_schema,
                force_grayscale=bool(preprocessing.get("force_grayscale", True)),
            )

    def loader(self, split: str) -> DataLoader:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        if split not in self.datasets:
            raise RuntimeError("Call setup() before requesting loaders")
        settings = self.config.get("loader", {})
        batch_size = int(settings.get("batch_size", 8))
        workers = int(settings.get("num_workers", 0))
        if batch_size < 1 or workers < 0:
            raise ValueError("batch_size must be positive and num_workers cannot be negative")
        sampling = str(settings.get("sampling", "shuffle"))
        if sampling not in {"shuffle", "weighted"}:
            raise ValueError("loader.sampling must be shuffle or weighted")
        seed = int(self.config.get("seed", 42)) + SPLIT_NAMES.index(split)
        generator = torch.Generator().manual_seed(seed)
        sampler = (
            make_weighted_sampler(self.frames[split], seed)
            if split == "fit_train" and sampling == "weighted"
            else None
        )
        return DataLoader(
            self.datasets[split],
            batch_size=batch_size,
            shuffle=split == "fit_train" and sampler is None,
            sampler=sampler,
            num_workers=workers,
            pin_memory=bool(settings.get("pin_memory", True)),
            persistent_workers=bool(settings.get("persistent_workers", False)) and workers > 0,
            drop_last=False,
            worker_init_fn=seed_loader_worker if workers > 0 else None,
            generator=generator,
        )

    def loaders(self) -> dict[str, DataLoader]:
        """Return one loader for each configured partition."""
        return {split: self.loader(split) for split in self.splits}


def smoke_test_loaders(
    loaders: dict[str, DataLoader],
    figure_path: Path,
    *,
    image_size: int,
    samples_per_split: int = 4,
) -> dict[str, dict[str, Any]]:
    """Load one batch per split, validate its contract, and save an anonymized grid."""
    missing = set(SPLIT_NAMES) - set(loaders)
    if missing:
        raise ValueError(f"Missing loaders: {sorted(missing)}")
    batches: dict[str, dict[str, Any]] = {}
    summary: dict[str, dict[str, Any]] = {}
    forbidden = {"absolute_path", "image_id", "record_id", "canonical_source_id"}
    for split in SPLIT_NAMES:
        batch = next(iter(loaders[split]))
        shape = list(batch["image"].shape)
        if shape[1:] != [3, image_size, image_size]:
            raise ValueError(f"Unexpected {split} image shape: {shape}")
        leaked_keys = sorted(forbidden & set(batch))
        if leaked_keys:
            raise ValueError(f"Predictive batch exposes source identifiers: {leaked_keys}")
        batches[split] = batch
        summary[split] = {
            "image_shape": shape,
            "mask_shape": list(batch["mask"].shape),
            "sample_valid": int(batch["valid_task_mask"]["sample"].sum()),
            "binary_valid": int(batch["valid_task_mask"]["binary"].sum()),
            "subtype_valid": int(batch["valid_task_mask"]["subtype"].sum()),
            "risk_valid": int(batch["valid_task_mask"]["risk"].sum()),
            "mask_valid": int(batch["valid_task_mask"]["mask"].sum()),
            "metadata_keys": sorted(batch["metadata"]),
            "forbidden_top_level_keys": leaked_keys,
        }

    columns = max(1, samples_per_split)
    figure, axes = plt.subplots(
        len(SPLIT_NAMES), columns, figsize=(3 * columns, 3 * len(SPLIT_NAMES))
    )
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for row_index, split in enumerate(SPLIT_NAMES):
        batch = batches[split]
        count = min(columns, len(batch["image"]))
        for column in range(columns):
            axis = axes[row_index, column]
            axis.axis("off")
            if column >= count:
                continue
            image = (batch["image"][column].cpu() * std + mean).clamp(0, 1)
            axis.imshow(image.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
            label = int(batch["binary_label"][column])
            valid = bool(batch["valid_task_mask"]["sample"][column])
            validity = "valid" if valid else "invalid"
            axis.set_title(f"{split} | class_{label} | {validity}", fontsize=8)
            if bool(batch["valid_task_mask"]["mask"][column]):
                mask = batch["mask"][column, 0].cpu().numpy()
                axis.imshow(mask, alpha=0.25, cmap="Greens", vmin=0, vmax=1)
    figure.suptitle("Phase 3 preprocessing batches (opaque sample identities)")
    figure.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = figure_path.with_name(f".{figure_path.name}.{uuid4().hex}.tmp.png")
    try:
        figure.savefig(temporary, dpi=150, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        plt.close(figure)
        os.replace(temporary, figure_path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return summary
