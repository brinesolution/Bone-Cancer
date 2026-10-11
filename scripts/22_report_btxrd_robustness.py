"""Write multi-seed and external-validation bookkeeping without inventing metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from bone_cancer_ai.evaluation.robustness import build_robustness_report
from bone_cancer_ai.optimization.contracts import write_json


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/btxrd_robustness.json")
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    report = build_robustness_report(
        project_root, seeds=tuple(args.seeds), external_root=args.external_root
    )
    output = args.output if args.output.is_absolute() else project_root / args.output
    write_json(output, report)
    print(f"Wrote BTXRD robustness report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
