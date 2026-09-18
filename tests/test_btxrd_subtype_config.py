from pathlib import Path

from bone_cancer_ai.models.hosp_bonenet import HOSPConfig
from bone_cancer_ai.utils.config import load_config


def test_btxrd_subtype_configuration_is_separate_and_masked() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = load_config(root / "configs/hosp_btxrd_subtype.yaml")
    schema = load_config(root / "configs/btxrd_label_schema.yaml")

    assert runtime["run_namespace"] == "experiments/btxrd_subtype"
    assert runtime["calibration"]["reuse_frozen_binary_test_metrics"] is False
    assert schema["subtype"]["enabled"] is True
    assert len(schema["subtype"]["classes"]) == 9
    model = HOSPConfig.from_mapping(runtime["model"])
    assert model.subtype_enabled is True
    assert model.num_subtypes == 9
