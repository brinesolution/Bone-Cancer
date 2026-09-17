"""Phase-gated inference configuration contracts."""

from pathlib import Path

import yaml


def test_inference_config_uses_phase9_frozen_outputs_for_phase10() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "inference.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["status"] == "pending_phase_10"
    assert config["evaluation_status"] == "phase_9_complete"
    assert config["checkpoint_alias"] == "hosp_bonenet.phase8.selected"
    assert config["checkpoint_registry"] == "artifacts/checkpoints/registry.json"
    assert "checkpoint" not in config
    assert config["binary_threshold"] == 0.5
    assert config["open_set_threshold"] is None
    assert config["conformal_calibration"] == "artifacts/metrics/hosp_conformal_calibration.json"
    assert config["metrics_path"] == "artifacts/metrics/final_model_comparison.csv"
