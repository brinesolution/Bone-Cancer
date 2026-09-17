"""Validate the sealed final model-comparison artifact without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_MODELS = {"DenseNet121", "EfficientNet-B3", "ConvNeXt-Tiny", "Swin-Tiny", "HOSP-BoneNet"}
REQUIRED_COLUMNS = {"model", "test_samples", "AUROC", "AUPRC", "F1", "checkpoint_sha256"}


def validate_comparison(project_root: Path, metrics_path: Path | None = None) -> dict[str, Any]:
    """Return metadata after validating the already-generated comparison CSV."""

    root = Path(project_root).resolve()
    path = Path(metrics_path or root / "artifacts" / "metrics" / "final_model_comparison.csv")
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"comparison artifact is missing columns: {sorted(missing)}")
    models = set(frame["model"].astype(str))
    if models != REQUIRED_MODELS:
        raise ValueError(f"comparison artifact models are incomplete: {sorted(models)}")
    if frame["model"].duplicated().any() or (frame["test_samples"] <= 0).any():
        raise ValueError("comparison artifact has duplicate models or invalid sample counts")
    for column in ("AUROC", "AUPRC", "F1"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"comparison artifact has invalid {column} values")
    return {
        "status": "complete",
        "path": path.as_posix(),
        "models": sorted(models),
        "rows": len(frame),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate_comparison(root, args.metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
