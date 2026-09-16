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


def publish_cache_index(
    index: pd.DataFrame,
    summary: dict[str, object],
    *,
    output_dir: Path,
    index_path: Path,
    summary_path: Path,
    prune_stale: bool,
) -> None:
    """Publish the new index before pruning files it no longer references."""
    atomic_write_csv(index, index_path)
    summary["pruned"] = 0
    if prune_stale and summary["failed"] == 0:
        expected = {Path(value).resolve() for value in index["cache_path"]}
        for candidate in output_dir.glob("*.png"):
            if candidate.resolve() not in expected:
                candidate.unlink()
                summary["pruned"] = int(summary["pruned"]) + 1
    atomic_write_json(summary, summary_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/data.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--manifest",
        type=Path,
        help="Explicit legacy/debug manifest; production uses fit_train and val.",
    )
    parser.add_argument("--split-dir", type=Path, default=Path("workspace/splits"))
    source.add_argument(
        "--splits",
        nargs="+",
        choices=("fit_train", "val"),
        help="Build from fit/validation split manifests.",
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
    if args.manifest is None and args.splits is None:
        args.splits = ["fit_train", "val"]
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        output_dir = paths.output_path(args.output_dir)
        index_path = paths.output_path(args.index_output)
        summary_path = paths.output_path(args.summary_output)
        if args.splits:
            if len(args.splits) != 2 or set(args.splits) != {"fit_train", "val"}:
                raise ValueError("production cache requires exactly --splits fit_train val")
            split_dir = paths.output_path(args.split_dir)
            frames = []
            for split in args.splits:
                split_path = split_dir / f"{split}.csv"
                frame = pd.read_csv(split_path, keep_default_na=True)
                if frame.empty or not frame["assigned_split"].eq(split).all():
                    raise ValueError(f"Invalid split manifest: {split_path}")
                frames.append(frame)
            manifest = pd.concat(frames, ignore_index=True)
        else:
            canonical_dir = paths.output_path("workspace/cache/resized")
            canonical_index = canonical_dir / "cache_index.csv"
            canonical_summary = paths.output_path("logs/cache_summary.json")
            if (
                output_dir == canonical_dir
                or index_path == canonical_index
                or summary_path == canonical_summary
            ):
                raise ValueError("legacy --manifest mode requires separate debug output paths")
            manifest_path = paths.output_path(args.manifest)
            manifest = pd.read_csv(manifest_path, keep_default_na=True)
        unsafe_sources = [
            value
            for value in manifest["absolute_path"]
            if not Path(str(value)).resolve().is_relative_to(paths.dataset_root)
        ]
        if unsafe_sources:
            raise ValueError("Cache manifest contains paths outside dataset_root")
        preprocessing = config.get("preprocessing", {})
        prior_index = (
            pd.read_csv(index_path, keep_default_na=False) if index_path.is_file() else None
        )
        index, summary = cache_manifest_images(
            manifest,
            output_dir,
            image_size=int(config.get("image_size", 384)),
            force_grayscale=bool(preprocessing.get("force_grayscale", True)),
            pad_value=int(preprocessing.get("pad_value", 0)),
            limit=args.limit,
            prior_index=prior_index,
        )
        summary["splits"] = list(args.splits or [])
        publish_cache_index(
            index,
            summary,
            output_dir=output_dir,
            index_path=index_path,
            summary_path=summary_path,
            prune_stale=bool(args.splits and args.limit is None),
        )
        print(json.dumps(summary, indent=2))
        logger.info("Cache build finished", extra={"event": "cache_build", "details": summary})
        return 0 if summary["failed"] == 0 else 2
    except Exception:
        logger.exception("Cache build failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
