"""Index the configured read-only dataset and write derived manifests."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from bone_cancer_ai.data.indexer import index_dataset
from bone_cancer_ai.data.quality import build_quality_report
from bone_cancer_ai.data.schema import atomic_write_csv, atomic_write_json
from bone_cancer_ai.utils.config import load_config
from bone_cancer_ai.utils.logging import configure_logging
from bone_cancer_ai.utils.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/data.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/manifests/dataset_master.csv")
    )
    parser.add_argument(
        "--quality-output", type=Path, default=Path("workspace/manifests/quality_report.csv")
    )
    parser.add_argument(
        "--issues-output", type=Path, default=Path("workspace/manifests/label_audit.csv")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("logs/dataset_index_summary.json")
    )
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = deepcopy(load_config(args.config))
        paths = ProjectPaths.from_config(config)
        if args.workers is not None:
            if args.workers < 1:
                raise ValueError("--workers must be positive")
            config.setdefault("audit", {})["workers"] = args.workers
        destinations = {
            "master": paths.output_path(args.output),
            "quality": paths.output_path(args.quality_output),
            "issues": paths.output_path(args.issues_output),
            "summary": paths.output_path(args.summary_output),
        }
        master, issues, summary = index_dataset(config)
        atomic_write_csv(master, destinations["master"])
        atomic_write_csv(build_quality_report(master), destinations["quality"])
        atomic_write_csv(issues, destinations["issues"])
        atomic_write_json(summary, destinations["summary"])
        print(json.dumps(summary, indent=2, default=str))
        logger.info(
            "Dataset indexing finished", extra={"event": "dataset_index", "details": summary}
        )
        return 0 if issues.empty else 2
    except Exception:
        logger.exception("Dataset indexing failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
