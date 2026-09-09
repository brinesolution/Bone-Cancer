"""Find exact and perceptual duplicates and report official-split leakage candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bone_cancer_ai.data.dedupe import (
    build_duplicate_groups,
    find_duplicate_edges,
    summarize_leakage,
)
from bone_cancer_ai.data.schema import atomic_write_csv, atomic_write_json
from bone_cancer_ai.utils.config import load_config
from bone_cancer_ai.utils.logging import configure_logging
from bone_cancer_ai.utils.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/data.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=Path("workspace/manifests/dataset_master.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/manifests/duplicate_groups.csv")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("logs/duplicate_summary.json")
    )
    parser.add_argument("--max-phash-distance", type=int)
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        manifest_path = paths.output_path(args.manifest)
        output_path = paths.output_path(args.output)
        summary_path = paths.output_path(args.summary_output)
        threshold = (
            args.max_phash_distance
            if args.max_phash_distance is not None
            else int(config.get("audit", {}).get("phash_distance", 4))
        )
        master = pd.read_csv(manifest_path, keep_default_na=True)
        edges = find_duplicate_edges(master, threshold)
        grouped = build_duplicate_groups(master, edges)
        summary = summarize_leakage(grouped)
        summary.update(
            {
                "max_phash_distance": threshold,
                "manifest": str(manifest_path),
                "report": str(output_path),
                "review_status": "Required before model training"
                if summary["cross_split_edges"]
                else "No cross-split duplicate evidence found",
            }
        )
        atomic_write_csv(grouped, output_path)
        atomic_write_json(summary, summary_path)
        print(json.dumps(summary, indent=2))
        logger.info(
            "Duplicate audit finished",
            extra={"event": "duplicate_audit", "details": summary},
        )
        return 0
    except Exception:
        logger.exception("Duplicate audit failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
