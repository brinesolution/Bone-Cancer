"""Overfit a fixed balanced fit-train subset to verify HOSP-BoneNet trainability."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from bone_cancer_ai.data.datamodule import BoneCancerDataModule
from bone_cancer_ai.data.dataset import BoneCancerDataset
from bone_cancer_ai.data.schema import atomic_write_json
from bone_cancer_ai.data.transforms import RadiographTransform
from bone_cancer_ai.losses import HOSPLoss, LossWeights
from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
from bone_cancer_ai.utils.config import load_config
from bone_cancer_ai.utils.paths import ProjectPaths
from bone_cancer_ai.utils.seed import seed_everything


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hosp_bonenet.yaml"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification/phase6_tiny_overfit.json"),
    )
    return parser.parse_args()


def _balanced_subset(frame: pd.DataFrame, samples: int) -> pd.DataFrame:
    if samples < 16 or samples > 32 or samples % 2:
        raise ValueError("samples must be an even value from 16 through 32")
    required = {"sample_id", "binary_label", "assigned_split", "is_eligible"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"fit manifest is missing columns: {sorted(missing)}")
    if not frame["assigned_split"].eq("fit_train").all():
        raise ValueError("overfit input must contain only fit_train rows")
    if not frame["is_eligible"].eq(True).all():
        raise ValueError("overfit input contains ineligible rows")
    per_class = samples // 2
    chosen = []
    for label in (0, 1):
        candidates = frame.loc[frame["binary_label"].eq(label)].sort_values("sample_id")
        if len(candidates) < per_class:
            raise ValueError(f"not enough fit_train samples for binary class {label}")
        chosen.append(candidates.head(per_class))
    return pd.concat(chosen, ignore_index=True).sort_values("sample_id").reset_index(drop=True)


def _evaluate(
    model: torch.nn.Module,
    criterion: HOSPLoss,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["binary_label"].to(device)
            valid = batch["valid_task_mask"]["binary"].to(device)
            sample_valid = batch["valid_task_mask"]["sample"].to(device)
            loss_batch = {
                "binary_label": targets,
                "valid_task_mask": {"binary": valid, "sample": sample_valid},
            }
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images)
                loss = criterion(output, loss_batch)["total"]
            valid = valid & sample_valid
            valid_count = int(valid.sum())
            loss_sum += float(loss) * valid_count
            correct += int(((output["binary_logits"] >= 0).long()[valid] == targets[valid]).sum())
            count += valid_count
    if count == 0:
        raise RuntimeError("overfit subset has no valid binary labels")
    return loss_sum / count, correct / count


def _preflight(dataset: BoneCancerDataset) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for index in range(len(dataset)):
        item = dataset[index]
        valid = item["valid_task_mask"]
        if not bool(valid["sample"]) or not bool(valid["binary"]):
            raise RuntimeError(f"selected overfit sample {index} failed runtime validation")
        label = int(item["binary_label"])
        if label not in (0, 1):
            raise RuntimeError(f"selected overfit sample {index} has an invalid binary label")
        counts[str(label)] += 1
    if counts["0"] != counts["1"] or sum(counts.values()) != len(dataset):
        raise RuntimeError("runtime-valid overfit samples must remain complete and balanced")
    return counts


def _implementation_fingerprint(
    project_root: Path, config_path: Path
) -> tuple[str, dict[str, str]]:
    roots = (
        project_root / "src" / "bone_cancer_ai" / "models" / "hosp_bonenet",
        project_root / "src" / "bone_cancer_ai" / "losses",
    )
    files = [config_path.resolve(), Path(__file__).resolve()]
    for root in roots:
        files.extend(sorted(root.glob("*.py")))
    hashes = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
    }
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), hashes


def main() -> int:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6 overfit verification requires CUDA")
    if args.image_size < 64 or args.batch_size < 1 or args.epochs < 5:
        raise ValueError("image size, batch size, and epochs are below verification limits")
    seed_everything(args.seed)
    config_path = args.config.resolve()
    config = load_config(config_path)
    data_config = load_config(config_path.parent / str(config["data_config"]))
    paths = ProjectPaths.from_config(data_config)
    split_dir = paths.output_path(
        Path(str(data_config.get("workspace_root", "workspace"))) / "splits"
    )
    schema_path = paths.output_path(str(data_config["schema_path"]))
    output_path = paths.output_path(args.output)
    label_schema = load_config(schema_path)
    data_module = BoneCancerDataModule(split_dir, data_config, label_schema)
    data_module.setup()
    subset = _balanced_subset(data_module.frames["fit_train"], args.samples)
    transform = RadiographTransform(
        args.image_size,
        train=False,
        seed=args.seed,
        augmentation={},
        normalize=True,
    )
    dataset = BoneCancerDataset(subset, transform, label_schema, force_grayscale=True)
    evaluated_class_counts = _preflight(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model_config = dict(config["model"])
    model_config.update({"pretrained": False, "dropout": 0.0})
    device = torch.device("cuda")
    model = create_hosp_bonenet(model_config, label_schema).to(device)
    criterion = HOSPLoss(LossWeights(binary=1.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    initial_loss, initial_accuracy = _evaluate(model, criterion, loader, device)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["binary_label"].to(device)
            loss_batch = {
                "binary_label": targets,
                "valid_task_mask": {
                    "binary": batch["valid_task_mask"]["binary"].to(device),
                    "sample": batch["valid_task_mask"]["sample"].to(device),
                },
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images)
                loss = criterion(output, loss_batch)["total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epoch_loss, epoch_accuracy = _evaluate(model, criterion, loader, device)
        history.append(
            {"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_accuracy}
        )
        if epoch >= 5 and epoch_loss <= initial_loss * 0.25 and epoch_accuracy >= 0.9375:
            break

    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = _evaluate(model, criterion, loader, device)
    loss_ratio = final_loss / initial_loss
    passed = loss_ratio <= 0.25 and final_accuracy >= 0.9375
    implementation_sha256, implementation_files = _implementation_fingerprint(
        paths.project_root, config_path
    )
    payload = {
        "status": "pass" if passed else "fail",
        "split": "fit_train",
        "split_generation_id": str(subset["split_generation_id"].iloc[0]),
        "samples": len(subset),
        "class_counts": {
            str(key): int(value)
            for key, value in subset["binary_label"].value_counts().sort_index().items()
        },
        "evaluated_samples": sum(evaluated_class_counts.values()),
        "evaluated_class_counts": evaluated_class_counts,
        "sample_ids": subset["sample_id"].tolist(),
        "backbone": model_config["backbone"],
        "pretrained": False,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_completed": len(history),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "thresholds": {"max_loss_ratio": 0.25, "min_accuracy": 0.9375},
        "history": history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": elapsed,
        "optional_supervision": "disabled",
        "implementation_sha256": implementation_sha256,
        "implementation_files_sha256": implementation_files,
    }
    atomic_write_json(payload, output_path)
    print(payload)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
