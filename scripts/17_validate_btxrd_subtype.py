"""Validate BTXRD subtype manifests and one masked HOSP-BoneNet pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bone_cancer_ai.data.datamodule import BoneCancerDataModule
from bone_cancer_ai.data.schema import atomic_write_json
from bone_cancer_ai.losses import HOSPLoss, LossWeights
from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
from bone_cancer_ai.utils.config import load_config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/hosp_btxrd_subtype.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--json-out", type=Path, default=Path("logs/btxrd_subtype_validation.json"))
    args = parser.parse_args()
    runtime = load_config(args.config)
    data_config = load_config(_resolve(root, runtime["data_config"]))
    schema = load_config(_resolve(root, runtime["label_schema"]))
    if not bool(schema.get("subtype", {}).get("enabled")):
        raise ValueError("BTXRD subtype schema must enable subtype supervision")
    if not bool(runtime.get("model", {}).get("subtype_enabled")):
        raise ValueError("BTXRD HOSP config must enable the subtype head")
    split_dir = _resolve(root, runtime["split_dir"])
    module = BoneCancerDataModule(split_dir, data_config, schema)
    module.setup()
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for BTXRD subtype validation but is unavailable")
    model_config = dict(runtime["model"])
    model_config["pretrained"] = False
    model = create_hosp_bonenet(model_config, schema).to(requested)
    criterion = HOSPLoss(LossWeights(**runtime.get("loss_weights", {})))
    batch = next(iter(module.loader("fit_train")))
    batch["image"] = batch["image"].to(requested)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(requested)
        elif isinstance(value, dict):
            batch[key] = {
                subkey: subvalue.to(requested) if isinstance(subvalue, torch.Tensor) else subvalue
                for subkey, subvalue in value.items()
            }
    model.train()
    predictions = model(batch["image"])
    losses = criterion(predictions, batch)
    losses["total"].backward()
    subtype_mask = batch["valid_task_mask"]["subtype"]
    result = {
        "status": "pass",
        "device": str(requested),
        "batch_size": int(batch["image"].shape[0]),
        "image_shape": list(batch["image"].shape),
        "subtype_logits_shape": list(predictions["subtype_logits"].shape),
        "subtype_valid_in_batch": int(subtype_mask.sum().item()),
        "loss_total": float(losses["total"].detach().cpu()),
        "loss_binary": float(losses["binary"].detach().cpu()),
        "loss_subtype": float(losses["subtype"].detach().cpu()),
        "split_dir": str(split_dir),
        "reuse_frozen_binary_test_metrics": False,
    }
    output = args.json_out if args.json_out.is_absolute() else root / args.json_out
    atomic_write_json(result, output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
