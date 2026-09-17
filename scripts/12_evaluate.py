"""Run or validate the frozen Phase 9 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bone_cancer_ai.evaluation.phase9 import finalize_phase9_visual_qa, run_phase9


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument(
        "--finalize-visual-qa",
        action="store_true",
        help="publish an awaiting run after visual_qa.json records a completed review",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    runner = finalize_phase9_visual_qa if args.finalize_visual_qa else run_phase9
    result = runner(project_root, args.config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
