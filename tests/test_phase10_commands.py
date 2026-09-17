"""Executable checks for the final comparison and explanation commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(project_root: Path, script: str, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_comparison_command_validates_five_frozen_models() -> None:
    root = Path(__file__).resolve().parents[1]
    result = _run(root, "13_compare_models.py")
    assert result["status"] == "complete"
    assert result["rows"] == 5


def test_explainability_command_validates_sealed_gallery() -> None:
    root = Path(__file__).resolve().parents[1]
    result = _run(root, "14_generate_explanations.py", "--split", "test")
    assert result["status"] == "complete"
    assert result["gallery_rows"] == 20
