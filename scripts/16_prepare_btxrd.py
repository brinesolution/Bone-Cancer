"""Audit BTXRD subtype metadata and emit immutable derived manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bone_cancer_ai.data.btxrd_migration import write_btxrd_artifacts
from bone_cancer_ai.utils.config import load_config


def _rooted(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/btxrd_data.yaml")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--main-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    project_root = Path(config.get("project_root", root))
    if not project_root.is_absolute():
        project_root = root / project_root
    source_root = args.source_root or _rooted(project_root, config["source_root"])
    main_manifest = args.main_manifest or _rooted(project_root, config["main_manifest"])
    output_root = args.output_root or _rooted(project_root, config["output_root"])
    summary = write_btxrd_artifacts(
        source_root,
        main_manifest_path=main_manifest,
        output_root=output_root,
        seed=int(config.get("seed", 42) if args.seed is None else args.seed),
        phash_distance=int(config.get("phash_distance", 4)),
        quality_thresholds=dict(config.get("quality_thresholds", {})),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    invalid = bool(summary["missing_images"] or summary["extra_image_files"])
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
