"""Build an all-record clean manifest using deterministic leakage/exclusion rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bone_cancer_ai.data.quality import build_clean_manifest, validate_clean_manifest
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
        "--duplicates", type=Path, default=Path("workspace/manifests/duplicate_groups.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/manifests/dataset_clean.csv")
    )
    parser.add_argument(
        "--exclusions", type=Path, default=Path("workspace/manifests/exclusions.csv")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("logs/clean_manifest_summary.json")
    )
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        manifest_path = paths.output_path(args.manifest)
        duplicate_path = paths.output_path(args.duplicates)
        output_path = paths.output_path(args.output)
        exclusion_path = paths.output_path(args.exclusions)
        summary_path = paths.output_path(args.summary_output)
        master = pd.read_csv(manifest_path, keep_default_na=True)
        duplicates = pd.read_csv(duplicate_path, keep_default_na=True)
        clean, exclusions, summary = build_clean_manifest(
            master, duplicates, config.get("split_policy", {})
        )
        validation = validate_clean_manifest(clean, set(master["record_id"].astype(str)))
        summary["validation"] = validation
        atomic_write_csv(clean, output_path)
        atomic_write_csv(exclusions, exclusion_path)
        atomic_write_json(summary, summary_path)
        print(json.dumps(summary, indent=2))
        ok = (
            validation["record_ids_unique"]
            and validation["complete_coverage"]
            and validation["group_pool_overlap"] == 0
            and summary["official_records_reassigned_to_training"] == 0
        )
        logger.info(
            "Clean manifest finished",
            extra={"event": "clean_manifest", "details": summary},
        )
        return 0 if ok else 2
    except Exception:
        logger.exception("Clean manifest failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
