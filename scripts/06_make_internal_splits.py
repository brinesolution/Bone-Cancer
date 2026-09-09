"""Create grouped fit/calibration manifests while preserving cleaned held-out sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bone_cancer_ai.data.datamodule import BoneCancerDataModule, smoke_test_loaders
from bone_cancer_ai.data.samplers import assign_internal_splits, validate_final_splits
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
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/splits"))
    parser.add_argument(
        "--summary-output", type=Path, default=Path("logs/internal_split_summary.json")
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("artifacts/figures/preprocessing_batch_grid.png"),
    )
    parser.add_argument(
        "--skip-loader-smoke",
        action="store_true",
        help="Skip real batch loading; intended only for isolated split tests.",
    )
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        manifest_path = paths.output_path(args.manifest)
        output_dir = paths.output_path(args.output_dir)
        summary_path = paths.output_path(args.summary_output)
        figure_path = paths.output_path(args.figure_output)
        clean = pd.read_csv(manifest_path, keep_default_na=True)
        policy = config.get("split_policy", {})
        assigned, summary = assign_internal_splits(
            clean,
            float(policy.get("calibration_fraction", 0.15)),
            seed=int(config.get("seed", 42)),
            search_trials=int(policy.get("calibration_search_trials", 128)),
        )
        validation = validate_final_splits(assigned)
        summary["validation"] = validation
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("fit_train", "calibration", "val", "test"):
            frame = assigned.loc[assigned["assigned_split"].eq(split)].reset_index(drop=True)
            atomic_write_csv(frame, output_dir / f"{split}.csv")
        if args.skip_loader_smoke:
            summary["loader_smoke"] = "skipped"
        else:
            schema_path = paths.output_path(config["schema_path"])
            label_schema = load_config(schema_path)
            module = BoneCancerDataModule(output_dir, config, label_schema)
            module.setup()
            summary["loader_smoke"] = smoke_test_loaders(
                module.loaders(), figure_path, image_size=int(config.get("image_size", 384))
            )
            summary["batch_grid"] = str(figure_path)
        atomic_write_json(summary, summary_path)
        print(json.dumps(summary, indent=2))
        ok = (
            validation["group_overlap"] == 0
            and validation["record_overlap"] == 0
            and validation["sample_overlap"] == 0
            and validation["generation_mismatch"] == 0
            and validation["unassigned_eligible_records"] == 0
            and validation["official_heldout_records_assigned_to_training"] == 0
        )
        logger.info(
            "Internal splits finished",
            extra={"event": "internal_splits", "details": summary},
        )
        return 0 if ok else 2
    except Exception:
        logger.exception("Internal split creation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
