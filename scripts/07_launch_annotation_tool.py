"""Validate or launch the local Phase 4 annotation interface."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bone_cancer_ai.annotations.context import load_annotation_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/annotation.yaml"))
    parser.add_argument(
        "--check", action="store_true", help="Check context without opening a server"
    )
    args = parser.parse_args()
    ctx = load_annotation_context(args.config)
    ctx.image_path(ctx.manifest.iloc[0].to_dict())
    if args.check:
        print(json.dumps(ctx.store.progress(len(ctx.manifest)), indent=2))
        return 0
    environment = dict(os.environ, BONE_ANNOTATION_CONFIG=str(args.config.resolve()))
    app = ctx.paths.project_root / "tools/annotation_app/app.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(ctx.config.get("port", 8502)),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ctx.paths.project_root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
