"""Optionally cache deterministic aspect-padded PNGs outside the immutable source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bone_cancer_ai.data.preprocessing import cache_manifest_images
from bone_cancer_ai.data.schema import atomic_write_csv, atomic_write_json
from bone_cancer_ai.utils.config import load_config
from bone_cancer_ai.utils.logging import configure_logging
from bone_cancer_ai.utils.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/data.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=Path("workspace/manifests/dataset_clean.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/cache/resized"))
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("workspace/cache/resized/cache_index.csv"),
    )
    parser.add_argument("--summary-output", type=Path, default=Path("logs/cache_summary.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        manifest_path = paths.output_path(args.manifest)
        output_dir = paths.output_path(args.output_dir)
        index_path = paths.output_path(args.index_output)
        summary_path = paths.output_path(args.summary_output)
        manifest = pd.read_csv(manifest_path, keep_default_na=True)
        unsafe_sources = [
            value
            for value in manifest["absolute_path"]
            if not Path(str(value)).resolve().is_relative_to(paths.dataset_root)
        ]
        if unsafe_sources:
            raise ValueError("Cache manifest contains paths outside dataset_root")
        preprocessing = config.get("preprocessing", {})
        index, summary = cache_manifest_images(
            manifest,
            output_dir,
            image_size=int(config.get("image_size", 384)),
            force_grayscale=bool(preprocessing.get("force_grayscale", True)),
            pad_value=int(preprocessing.get("pad_value", 0)),
            limit=args.limit,
        )
        atomic_write_csv(index, index_path)
        atomic_write_json(summary, summary_path)
        print(json.dumps(summary, indent=2))
        logger.info("Cache build finished", extra={"event": "cache_build", "details": summary})
        return 0 if summary["failed"] == 0 else 2
    except Exception:
        logger.exception("Cache build failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
