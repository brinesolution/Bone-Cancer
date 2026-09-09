"""Verify labels with local provenance and render required dataset-audit figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bone_cancer_ai.data.audit import audit_labels, save_audit_figures
from bone_cancer_ai.data.provenance import load_local_provenance
from bone_cancer_ai.data.schema import ISSUE_COLUMNS, atomic_write_csv, atomic_write_json
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
    parser.add_argument("--issues", type=Path, default=Path("workspace/manifests/label_audit.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument(
        "--summary-output", type=Path, default=Path("logs/dataset_audit_summary.json")
    )
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        manifest_path = paths.output_path(args.manifest)
        issues_path = paths.output_path(args.issues)
        figure_dir = paths.output_path(args.figure_dir)
        summary_path = paths.output_path(args.summary_output)
        master = pd.read_csv(manifest_path, keep_default_na=True)
        existing_issues = (
            pd.read_csv(issues_path, keep_default_na=False)
            if issues_path.is_file() and issues_path.stat().st_size > 0
            else pd.DataFrame(columns=ISSUE_COLUMNS)
        )
        provenance = load_local_provenance(config, paths.project_root)
        enriched, issues, summary = audit_labels(master, provenance, existing_issues)
        dimensions = enriched.dropna(subset=["width", "height"])
        class_counts_by_split: dict[str, dict[str, int]] = {}
        for (split, label), count in enriched.groupby(
            ["official_split", "binary_label"], sort=True
        ).size().items():
            class_counts_by_split.setdefault(str(split), {})[str(int(label))] = int(count)
        summary.update(
            {
                "label_csv_columns": ["image_id", "binary_label"],
                "quality_counts": {
                    str(key): int(value)
                    for key, value in enriched["quality_status"].value_counts().items()
                },
                "extension_counts": {
                    str(key): int(value)
                    for key, value in enriched["extension"].value_counts().items()
                },
                "mode_counts": {
                    str(key): int(value)
                    for key, value in enriched["pil_mode"].value_counts().items()
                },
                "channel_counts": {
                    str(int(key)): int(value)
                    for key, value in enriched["channels"]
                    .dropna()
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "dimension_summary": {
                    "width": {
                        "min": int(dimensions["width"].min()),
                        "median": float(dimensions["width"].median()),
                        "max": int(dimensions["width"].max()),
                    },
                    "height": {
                        "min": int(dimensions["height"].min()),
                        "median": float(dimensions["height"].median()),
                        "max": int(dimensions["height"].max()),
                    },
                    "unique_width_height_pairs": int(
                        dimensions[["width", "height"]].drop_duplicates().shape[0]
                    ),
                },
                "class_counts_by_split": class_counts_by_split,
                "rgb_encoded_grayscale_images": int(
                    enriched["is_rgb_encoded_grayscale"].fillna(False).sum()
                ),
                "subtype_counts": {
                    str(key): int(value)
                    for key, value in enriched.loc[
                        enriched["subtype_label"].fillna("").ne(""), "subtype_label"
                    ]
                    .value_counts()
                    .items()
                },
                "risk_counts": {
                    str(key): int(value)
                    for key, value in enriched.loc[
                        enriched["risk_label"].fillna("").ne(""), "risk_label"
                    ]
                    .value_counts()
                    .items()
                },
            }
        )
        outputs = save_audit_figures(
            enriched,
            figure_dir,
            seed=int(config.get("seed", 42)),
            sample_size=int(config.get("audit", {}).get("sample_grid_size", 16)),
        )
        summary["figure_paths"] = [str(path) for path in outputs]
        atomic_write_csv(enriched, manifest_path)
        atomic_write_csv(issues, issues_path)
        atomic_write_json(summary, summary_path)
        print(json.dumps(summary, indent=2, default=str))
        logger.info("Dataset audit finished", extra={"event": "dataset_audit", "details": summary})
        return 0 if summary["binary_semantics_verified"] else 2
    except Exception:
        logger.exception("Dataset audit failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
