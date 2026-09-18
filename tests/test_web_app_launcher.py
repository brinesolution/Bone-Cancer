from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.streamlit_app import load_model_comparison
from launch_web_app import build_command


def test_launcher_uses_project_interpreter_and_streamlit_entrypoint() -> None:
    root = Path(__file__).resolve().parents[1]
    command = build_command(root, host="127.0.0.1", port=8765, headless=True)

    assert command[:4] == [str(root / ".venv" / "Scripts" / "python.exe"), "-m", "streamlit", "run"]
    assert str(root / "app" / "streamlit_app.py") in command
    assert "--server.port" in command
    assert "8765" in command


def test_model_comparison_loader_returns_sealed_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = load_model_comparison(str(root))

    assert isinstance(frame, pd.DataFrame)
    assert set(frame["model"]) == {
        "DenseNet121",
        "EfficientNet-B3",
        "ConvNeXt-Tiny",
        "Swin-Tiny",
        "HOSP-BoneNet",
    }
    assert len(frame) == 5
    assert frame["test_samples"].eq(1882).all()


def test_streamlit_app_exposes_image_and_results_views() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app" / "streamlit_app.py")).run(timeout=60)

    assert not app.exception
    assert len(app.tabs) == 2
    assert len(app.dataframe) == 1
    assert len(app.metric) == 3


def test_streamlit_app_exposes_only_current_btxrd_inference_action() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "app" / "streamlit_app.py").read_text(encoding="utf-8")

    assert "Run frozen HOSP-BoneNet" not in source
    assert "Run current BTXRD subtype HOSP-BoneNet" in source
