"""Command exit status must reflect validation failures."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_validation_command_rejects_corrupt_source_report(tmp_path, monkeypatch):
    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts/08_validate_annotations.py"))[
        "main"
    ]
    (tmp_path / "source_annotations.csv").touch()
    ctx = SimpleNamespace(
        root=tmp_path,
        manifest=[],
        paths=SimpleNamespace(output_path=lambda value: tmp_path / "report.json"),
        store=SimpleNamespace(progress=lambda count: {}),
    )
    monkeypatch.setattr(sys, "argv", ["validate"])
    for name, function in {
        "load_annotation_context": lambda path: ctx,
        "validate_project_annotations": lambda context: {"status": "passed"},
        "validate_source_import": lambda path: {"valid": False, "issues": ["corrupt mask"]},
        "export_annotations": lambda context: pd.DataFrame(),
    }.items():
        monkeypatch.setitem(main.__globals__, name, function)
    assert main() == 1
