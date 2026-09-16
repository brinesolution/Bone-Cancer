"""Validate existing Phase 5/7 selections and refresh their checkpoint aliases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "09_train_baseline.py"
HOSP_SCRIPT = PROJECT_ROOT / "scripts" / "10_train_hosp_bonenet.py"


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def refresh_checkpoint_registry(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Register all existing validated final selections without running training."""

    root = Path(project_root).resolve()
    baseline = _load_script(root / "scripts" / BASELINE_SCRIPT.name, "phase5_baseline")
    hosp = _load_script(root / "scripts" / HOSP_SCRIPT.name, "phase7_hosp")
    phase5 = baseline.refresh_phase5_registry(root)
    phase7 = hosp.refresh_phase7_registry(root)
    aliases = sorted([*phase5["aliases"], phase7["alias"]])
    return {
        "aliases": aliases,
        "phase5_models": phase5["models"],
        "phase7_run": phase7["run"],
        "status": "valid",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(refresh_checkpoint_registry(args.project_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
