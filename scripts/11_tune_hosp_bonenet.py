"""Run resumable validation-only Phase 8 optimization for HOSP-BoneNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bone_cancer_ai.optimization.contracts import (
    ablation_model_config,
    load_completed_stage,
    resolve_phase8_config,
    sha256_file,
)
from bone_cancer_ai.optimization.workflow import PROJECT_ROOT, run_phase8

__all__ = [
    "ablation_model_config",
    "load_completed_stage",
    "resolve_phase8_config",
    "run_phase8",
    "sha256_file",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "optimization.yaml"
    )
    parser.add_argument(
        "--force-new-run",
        action="store_true",
        help="start a new immutable experiment instead of resuming the fingerprinted run",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    result = run_phase8(arguments.config, force_new_run=arguments.force_new_run)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
