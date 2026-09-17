"""Run frozen HOSP-BoneNet inference on one local radiograph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bone_cancer_ai.inference.postprocess import result_to_json
from bone_cancer_ai.inference.predictor import FrozenPredictor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device) if args.device else None
    predictor = FrozenPredictor.from_project_root(root, args.config, device=device)
    payload = result_to_json(predictor.predict_path(args.image))
    if args.json_out is not None:
        destination = args.json_out if args.json_out.is_absolute() else root / args.json_out
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
