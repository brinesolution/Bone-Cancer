"""Phase-gated inference configuration contracts."""

from pathlib import Path

import yaml


def test_inference_config_uses_verified_hosp_alias_while_phase9_remains_pending() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "inference.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["status"] == "pending_phase_10"
    assert config["evaluation_status"] == "pending_phase_9"
    assert config["checkpoint_alias"] == "hosp_bonenet.phase7.selected"
    assert config["checkpoint_registry"] == "artifacts/checkpoints/registry.json"
    assert "checkpoint" not in config
    assert config["binary_threshold"] is None
    assert config["open_set_threshold"] is None
    assert config["conformal_calibration"] is None
