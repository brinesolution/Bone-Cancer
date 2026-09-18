"""Launch the project Streamlit application with the pinned virtual environment."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_command(project_root: Path, *, host: str, port: int, headless: bool) -> list[str]:
    """Build the subprocess command for the pinned Streamlit entry point."""

    root = Path(project_root).resolve()
    interpreter = root / ".venv" / "Scripts" / "python.exe"
    app = root / "app" / "streamlit_app.py"
    if not interpreter.is_file():
        raise FileNotFoundError(f"project interpreter missing: {interpreter}")
    if not app.is_file():
        raise FileNotFoundError(f"Streamlit entry point missing: {app}")
    return [
        str(interpreter),
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8501)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    command = build_command(root, host=args.host, port=args.port, headless=args.headless)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    try:
        return subprocess.call(command, cwd=root, env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
