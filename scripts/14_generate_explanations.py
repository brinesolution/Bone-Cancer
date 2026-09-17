"""Validate the sealed Phase 9 explanation gallery without rerunning test inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def validate_explanations(project_root: Path, split: str = "test") -> dict[str, Any]:
    """Return metadata after validating the generated gallery and visual-QA record."""

    if split != "test":
        raise ValueError("only the sealed test explanation gallery is available")
    root = Path(project_root).resolve()
    output = root / "artifacts" / "explainability" / "phase9"
    index_path = output / "gallery_index.csv"
    qa_path = output / "visual_qa.json"
    failure_path = output / "failure_gallery.png"
    if not index_path.is_file() or not qa_path.is_file() or not failure_path.is_file():
        raise FileNotFoundError("sealed Phase 9 explanation outputs are incomplete")
    index = pd.read_csv(index_path)
    if index.empty or not {"category", "sample_id", "prediction"}.issubset(index.columns):
        raise ValueError("gallery index is empty or missing required columns")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa.get("status") not in {"complete", "complete_with_limitations"}:
        raise ValueError(f"visual QA is not complete: {qa.get('status')}")
    return {
        "status": "complete",
        "path": output.as_posix(),
        "split": split,
        "gallery_rows": len(index),
        "visual_qa_status": qa["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test"], default="test")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate_explanations(root, args.split), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
