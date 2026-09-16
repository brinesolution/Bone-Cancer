"""Contracts for rebuilding selected checkpoint aliases without training."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "refresh_checkpoint_registry.py"


def test_refresh_command_combines_four_phase5_and_one_phase7_selection(
    monkeypatch, tmp_path: Path
) -> None:
    spec = importlib.util.spec_from_file_location("artifact_refresh", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    refresh_cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = refresh_cli
    spec.loader.exec_module(refresh_cli)
    calls: list[tuple[str, Path]] = []
    baseline = SimpleNamespace(
        refresh_phase5_registry=lambda root: (
            calls.append(("phase5", root))
            or {
                "aliases": [
                    "baseline.convnext_tiny.phase5.selected",
                    "baseline.densenet121.phase5.selected",
                    "baseline.efficientnet_b3.phase5.selected",
                    "baseline.swin_tiny.phase5.selected",
                ],
                "models": 4,
                "status": "valid",
            }
        )
    )
    hosp = SimpleNamespace(
        refresh_phase7_registry=lambda root: (
            calls.append(("phase7", root))
            or {
                "alias": "hosp_bonenet.phase7.selected",
                "run": "experiments/hosp_bonenet/run-001",
                "status": "valid",
            }
        )
    )
    monkeypatch.setattr(
        refresh_cli,
        "_load_script",
        lambda path, name: baseline if name == "phase5_baseline" else hosp,
    )

    result = refresh_cli.refresh_checkpoint_registry(tmp_path)

    assert calls == [("phase5", tmp_path.resolve()), ("phase7", tmp_path.resolve())]
    assert result == {
        "aliases": [
            "baseline.convnext_tiny.phase5.selected",
            "baseline.densenet121.phase5.selected",
            "baseline.efficientnet_b3.phase5.selected",
            "baseline.swin_tiny.phase5.selected",
            "hosp_bonenet.phase7.selected",
        ],
        "phase5_models": 4,
        "phase7_run": "experiments/hosp_bonenet/run-001",
        "status": "valid",
    }
