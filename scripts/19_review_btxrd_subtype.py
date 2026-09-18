"""Write a concise integrity review for the completed BTXRD subtype run."""

from __future__ import annotations

import argparse
from pathlib import Path

from bone_cancer_ai.evaluation.btxrd_review import review_btxrd_run
from bone_cancer_ai.optimization.contracts import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/btxrd_subtype_review.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    review = review_btxrd_run(root)
    write_json(output, review)
    print(f"BTXRD review PASS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
