"""Fit calibration temperatures on BTXRD calibration rows and write derived metrics."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bone_cancer_ai.artifacts.registry import resolve_checkpoint
from bone_cancer_ai.evaluation.calibration import BinaryTemperatureScaler, TemperatureScaler
from bone_cancer_ai.evaluation.classification import binary_classification_metrics
from bone_cancer_ai.evaluation.hosp_btxrd import (
    MulticlassConformalCalibrator,
    subtype_classification_metrics,
)
from bone_cancer_ai.inference.conformal import BinaryConformalCalibrator
from bone_cancer_ai.optimization.contracts import write_json
from bone_cancer_ai.utils.config import load_config


def _list_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = [ast.literal_eval(str(value)) for value in frame[column].tolist()]
    except (KeyError, SyntaxError, ValueError) as error:
        raise ValueError(f"{column} is not a serialized list column") from error
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics/btxrd_subtype_temperature.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    run_dir = args.run_dir or root / "experiments/btxrd_subtype/run-a8ae625f34c1"
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    calibration_binary = pd.read_csv(run_dir / "calibration_binary.csv")
    test_binary = pd.read_csv(run_dir / "test_binary.csv")
    calibration_subtype = pd.read_csv(run_dir / "calibration_subtype.csv")
    test_subtype = pd.read_csv(run_dir / "test_subtype.csv")
    binary_scaler = BinaryTemperatureScaler.fit(
        calibration_binary["logit"].to_numpy(), calibration_binary["target"].to_numpy()
    )
    subtype_scaler = TemperatureScaler.fit(
        _list_column(calibration_subtype, "logits"), calibration_subtype["target"].to_numpy()
    )
    calibration_binary_logits_np = calibration_binary["logit"].to_numpy()
    calibration_binary_targets_np = calibration_binary["target"].to_numpy()
    calibration_subtype_logits_np = _list_column(calibration_subtype, "logits")
    calibration_subtype_targets_np = calibration_subtype["target"].to_numpy()
    cal_binary_logits = torch.tensor(calibration_binary["logit"].to_numpy(), dtype=torch.float64)
    test_binary_logits = torch.tensor(test_binary["logit"].to_numpy(), dtype=torch.float64)
    cal_subtype_logits = torch.tensor(_list_column(calibration_subtype, "logits"))
    test_subtype_logits = torch.tensor(_list_column(test_subtype, "logits"))
    scaled_cal_binary = binary_scaler.transform_logits(cal_binary_logits).numpy()
    scaled_test_binary = binary_scaler.transform_logits(test_binary_logits).numpy()
    scaled_cal_subtype = subtype_scaler.transform_logits(cal_subtype_logits)
    scaled_test_subtype = subtype_scaler.transform_logits(test_subtype_logits)
    binary_calibrator = BinaryConformalCalibrator.fit(
        torch.sigmoid(torch.tensor(scaled_cal_binary)).numpy(),
        calibration_binary["target"].to_numpy(),
    )
    subtype_probabilities = torch.softmax(scaled_cal_subtype, dim=1).numpy()
    test_subtype_probabilities = torch.softmax(scaled_test_subtype, dim=1).numpy()
    subtype_calibrator = MulticlassConformalCalibrator.fit(
        subtype_probabilities, calibration_subtype["target"].to_numpy()
    )
    schema = load_config(root / "configs/btxrd_label_schema.yaml")
    class_names = tuple(schema["subtype"]["classes"])
    evaluation = {
        "binary": binary_classification_metrics(scaled_test_binary, test_binary["target"]),
        "binary_conformal": binary_calibrator.summarize(
            torch.sigmoid(torch.tensor(scaled_test_binary)).numpy(), test_binary["target"]
        ),
        "subtype": subtype_classification_metrics(
            scaled_test_subtype, test_subtype["target"], class_names=class_names
        ),
        "subtype_conformal": subtype_calibrator.summarize(
            test_subtype_probabilities, test_subtype["target"]
        ),
        "subtype_calibrator": subtype_calibrator.state_dict(),
    }
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    record = resolve_checkpoint(root, "hosp_bonenet.btxrd.subtype.selected")
    result = {
        "status": "complete",
        "protocol": "btxrd_calibration_temperature_scaling",
        "source_run": str(run_dir.relative_to(root)).replace("\\", "/"),
        "checkpoint_sha256": record.checkpoint_sha256,
        "configuration_fingerprint": record.configuration_fingerprint,
        "source_protocol": metrics.get("protocol"),
        "calibration_rows": int(len(calibration_binary)),
        "test_rows": int(len(test_binary)),
        "binary_temperature": binary_scaler.to_dict(),
        "subtype_temperature": subtype_scaler.to_dict(),
        "calibration_nll": {
            "binary_before": binary_scaler.nll(
                calibration_binary_logits_np, calibration_binary_targets_np, temperature=1.0
            ),
            "binary_after": binary_scaler.nll(
                calibration_binary_logits_np, calibration_binary_targets_np
            ),
            "subtype_before": subtype_scaler.nll(
                calibration_subtype_logits_np, calibration_subtype_targets_np, temperature=1.0
            ),
            "subtype_after": subtype_scaler.nll(
                calibration_subtype_logits_np, calibration_subtype_targets_np
            ),
        },
        "evaluation": evaluation,
        "source_metrics_unchanged": True,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    write_json(output, result)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
