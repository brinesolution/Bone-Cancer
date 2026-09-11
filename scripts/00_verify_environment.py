"""Verify the installed environment and configured source paths without training."""

import argparse
import json
import os
from pathlib import Path

from bone_cancer_ai.utils.config import load_config
from bone_cancer_ai.utils.logging import configure_logging
from bone_cancer_ai.utils.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "data.yaml",
    )
    parser.add_argument("--output", type=Path, help="New JSON report path relative to project_root")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")
    parser.add_argument(
        "--skip-tensor-check",
        action="store_true",
        help="Verify dependency imports without allocating a CPU/CUDA tensor",
    )
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        paths = ProjectPaths.from_config(config)
        output = paths.output_path(args.output) if args.output is not None else None
        if not paths.project_root.is_dir() or not paths.dataset_root.is_dir():
            raise FileNotFoundError("Configured project/source directory does not exist")
        splits = config["official_splits"]
        if not isinstance(splits, list) or not splits:
            raise ValueError("official_splits must be a nonempty list")
        for split in splits:
            if not isinstance(split, str) or Path(split).name != split or split in {".", ".."}:
                raise ValueError(f"Invalid source split name: {split!r}")
            if not (paths.dataset_root / split).is_dir():
                raise FileNotFoundError(f"Missing source split directory: {split}")
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        from bone_cancer_ai.utils.environment import environment_report

        report = environment_report(
            require_cuda=args.require_cuda,
            run_tensor_check=not args.skip_tensor_check,
        )
        report["paths"] = {
            "project_root": str(paths.project_root),
            "dataset_root": str(paths.dataset_root),
            "official_splits": splits,
            "source_read_only": True,
        }
        serialized = json.dumps(report, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        print(serialized)
        logger.info(
            "Environment verification finished",
            extra={
                "event": "environment",
                "details": {
                    "ok": report["ok"],
                    "cuda_available": report["cuda"]["available"],
                },
            },
        )
        return 0 if report["ok"] else 1
    except Exception:
        logger.exception("Environment verification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
