"""Import source LabelMe annotations or validate and export local annotation records."""

import argparse
import json
from pathlib import Path

from bone_cancer_ai.annotations.context import load_annotation_context
from bone_cancer_ai.annotations.source_import import (
    import_source_annotations,
    validate_source_import,
)
from bone_cancer_ai.annotations.validation import validate_project_annotations
from bone_cancer_ai.annotations.workflow import export_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument(
        "--import-source",
        action="store_true",
        help="Create a new immutable BTXRD spatial conversion generation",
    )
    args = parser.parse_args()
    ctx = load_annotation_context(args.config)
    report = {"project": validate_project_annotations(ctx)}
    if args.import_source:
        report["import"] = import_source_annotations(args.config)
    if (ctx.root / "source_annotations.csv").is_file():
        report["source_validation"] = validate_source_import(args.config)
    else:
        report["source_validation"] = {"status": "Not run"}
    export_annotations(ctx).to_csv(ctx.root / "image_labels.csv", index=False)
    report["progress"] = ctx.store.progress(len(ctx.manifest))
    destination = ctx.paths.output_path("logs/phase4_annotation_validation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["source_validation"].get("valid") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
