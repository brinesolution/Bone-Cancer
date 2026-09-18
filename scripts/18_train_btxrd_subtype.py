"""Train and freshly evaluate subtype-enabled HOSP-BoneNet on BTXRD artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from bone_cancer_ai.artifacts.registry import promote_checkpoint, resolve_checkpoint
from bone_cancer_ai.data.datamodule import BoneCancerDataModule
from bone_cancer_ai.data.schema import atomic_write_json
from bone_cancer_ai.evaluation.hosp_btxrd import predict_hosp_multitask
from bone_cancer_ai.models.hosp_bonenet import create_hosp_bonenet
from bone_cancer_ai.training.hosp_trainer import HOSPMultiTaskTrainer
from bone_cancer_ai.utils.config import load_config


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_status(path: Path, status: dict[str, Any]) -> None:
    atomic_write_json(status, path)


def _prediction_arrays(frame: pd.DataFrame, *, subtype: bool = False) -> tuple[Any, Any]:
    if frame.empty:
        raise ValueError("prediction frame is empty")
    if subtype:
        logits = torch.tensor(frame["logits"].tolist(), dtype=torch.float32)
    else:
        logits = torch.tensor(frame["logit"].tolist(), dtype=torch.float32)
    targets = torch.tensor(frame["target"].tolist(), dtype=torch.long)
    return logits, targets


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/hosp_btxrd_subtype.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--epochs", type=int, default=None, help="explicit run override; do not promote overrides"
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="limit each split for a smoke run"
    )
    parser.add_argument("--allow-cpu", action="store_true", help="permit CPU exploratory runs")
    parser.add_argument("--force-new-run", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    runtime = load_config(args.config)
    if runtime.get("status") != "phase_b_pending_training":
        raise ValueError("BTXRD runtime config is not the Phase B contract")
    if runtime.get("run_namespace") != "experiments/btxrd_subtype":
        raise ValueError("BTXRD training must use its isolated run namespace")
    data_config = load_config(_resolve(root, runtime["data_config"]))
    schema = load_config(_resolve(root, runtime["label_schema"]))
    if not bool(schema.get("subtype", {}).get("enabled")) or not bool(
        runtime["model"].get("subtype_enabled")
    ):
        raise ValueError(
            "Phase B requires verified subtype supervision and an enabled subtype head"
        )
    split_dir = _resolve(root, runtime["split_dir"])
    split_paths = {
        name: split_dir / f"{name}.csv" for name in ("fit_train", "calibration", "val", "test")
    }
    if any(not path.is_file() for path in split_paths.values()):
        raise FileNotFoundError("BTXRD split artifacts are incomplete")

    training = dict(runtime.get("training", {}))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        training["epochs"] = args.epochs
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    requested = torch.device(args.device or training.get("device", "cuda"))
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the Phase B config; use --allow-cpu with --device "
            "cpu for an exploratory run"
        )
    if (
        requested.type == "cpu"
        and not args.allow_cpu
        and str(training.get("device", "cuda")) == "cuda"
    ):
        raise RuntimeError(
            "CPU training is disabled by the Phase B config; pass --allow-cpu explicitly"
        )

    loader_config = dict(data_config)
    loader_config["loader"] = dict(loader_config.get("loader", {}))
    loader_config["loader"]["batch_size"] = int(
        training.get("batch_size", loader_config["loader"].get("batch_size", 2))
    )
    loader_config["loader"]["num_workers"] = int(
        training.get("num_workers", loader_config["loader"].get("num_workers", 0))
    )
    module = BoneCancerDataModule(split_dir, loader_config, schema)
    module.setup()
    loaders = module.loaders()
    if args.max_batches is not None:

        def limited(loader: Any) -> Any:
            for index, batch in enumerate(loader):
                if index >= args.max_batches:
                    break
                yield batch

        loaders = {name: limited(loader) for name, loader in loaders.items()}

    contract = {
        "protocol": "btxrd_phase_b_subtype_enabled",
        "runtime": runtime,
        "data_config": loader_config,
        "label_schema": schema,
        "split_sha256": {name: _sha256(path) for name, path in split_paths.items()},
        "no_frozen_binary_test_reuse": True,
        "execution": {
            "device": str(requested),
            "epochs_override": args.epochs,
            "max_batches": args.max_batches,
            "no_promote": bool(args.no_promote),
        },
    }
    fingerprint = _fingerprint(contract)
    run_id = f"run-{fingerprint[:12]}"
    if args.force_new_run:
        run_id += "-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_dir = root / runtime["run_namespace"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    if status_path.is_file() and not args.force_new_run:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        reusable_artifacts = (run_dir / "best.pt").is_file() and metrics_path.is_file()
        try:
            selected = resolve_checkpoint(root, "hosp_bonenet.btxrd.subtype.selected")
            reusable_artifacts = reusable_artifacts and (
                selected.configuration_fingerprint == fingerprint
                and selected.source_run == str(run_dir.relative_to(root)).replace("\\", "/")
            )
        except (FileNotFoundError, ValueError):
            reusable_artifacts = False
        if (
            previous.get("status") == "complete"
            and previous.get("configuration_fingerprint") == fingerprint
            and bool(previous.get("reusable", False))
            and reusable_artifacts
        ):
            print(
                json.dumps(
                    {"status": "reused", "run_dir": str(run_dir), "metrics": str(metrics_path)},
                    indent=2,
                )
            )
            return 0
    atomic_write_json(contract, run_dir / "contract.json")
    reusable = args.epochs is None and args.max_batches is None and not args.no_promote
    _write_status(
        status_path,
        {
            "status": "training",
            "configuration_fingerprint": fingerprint,
            "run_dir": str(run_dir),
            "reusable": False,
        },
    )

    trainer_config = dict(training)
    trainer_config["loss_weights"] = dict(runtime.get("loss_weights", {}))
    trainer_config["selection_metric"] = "AUROC"
    alias = (
        "hosp_bonenet.btxrd.subtype.test"
        if args.no_promote or args.epochs is not None or args.max_batches is not None
        else "hosp_bonenet.btxrd.subtype"
    )
    model = create_hosp_bonenet(runtime["model"], schema)
    resume = run_dir / "last.pt"
    trainer = HOSPMultiTaskTrainer(
        model,
        model_alias=alias,
        resolved_config=trainer_config,
        checkpoint_dir=run_dir,
        device=requested,
        resume_from=resume if resume.is_file() and not args.force_new_run else None,
    )
    fit_result = trainer.fit(loaders["fit_train"], loaders["val"])

    best_model = create_hosp_bonenet(runtime["model"], schema)
    best_trainer = HOSPMultiTaskTrainer(
        best_model,
        model_alias=alias,
        resolved_config=trainer_config,
        checkpoint_dir=run_dir,
        device=requested,
        resume_from=Path(fit_result["best_checkpoint"]),
    )
    calibration = predict_hosp_multitask(
        best_trainer.model,
        loaders["calibration"],
        device=requested,
        amp=bool(training.get("amp", True)),
    )
    test = predict_hosp_multitask(
        best_trainer.model, loaders["test"], device=requested, amp=bool(training.get("amp", True))
    )
    calibration["binary"].to_csv(run_dir / "calibration_binary.csv", index=False)
    calibration["subtype"].to_csv(run_dir / "calibration_subtype.csv", index=False)
    test["binary"].to_csv(run_dir / "test_binary.csv", index=False)
    test["subtype"].to_csv(run_dir / "test_subtype.csv", index=False)

    from bone_cancer_ai.evaluation.hosp_btxrd import evaluate_hosp_btxrd_predictions

    cal_binary_logits, cal_binary_targets = _prediction_arrays(calibration["binary"])
    test_binary_logits, test_binary_targets = _prediction_arrays(test["binary"])
    cal_subtype_logits, cal_subtype_targets = _prediction_arrays(
        calibration["subtype"], subtype=True
    )
    test_subtype_logits, test_subtype_targets = _prediction_arrays(test["subtype"], subtype=True)
    evaluation = evaluate_hosp_btxrd_predictions(
        calibration_binary_logits=cal_binary_logits,
        calibration_binary_targets=cal_binary_targets,
        test_binary_logits=test_binary_logits,
        test_binary_targets=test_binary_targets,
        calibration_subtype_probabilities=torch.softmax(cal_subtype_logits, dim=1).numpy(),
        calibration_subtype_targets=cal_subtype_targets,
        test_subtype_logits=test_subtype_logits,
        test_subtype_targets=test_subtype_targets,
        alpha_binary=float(runtime["evaluation"]["alpha_binary"]),
        alpha_subtype=float(runtime["evaluation"]["alpha_subtype"]),
    )
    metrics = {
        "protocol": "btxrd_fresh_calibration_and_test",
        "configuration_fingerprint": fingerprint,
        "run_dir": str(run_dir.relative_to(root)),
        "fit": fit_result,
        "evaluation": evaluation,
        "calibration_rows": int(len(calibration["binary"])),
        "test_rows": int(len(test["binary"])),
        "subtype_calibration_rows": int(len(calibration["subtype"])),
        "subtype_test_rows": int(len(test["subtype"])),
        "rejected_calibration": int(calibration["rejected_count"]),
        "rejected_test": int(test["rejected_count"]),
        "reused_frozen_binary_test_metrics": False,
    }
    atomic_write_json(metrics, metrics_path)
    if not args.no_promote and args.epochs is None and args.max_batches is None:
        promote_checkpoint(
            root,
            alias="hosp_bonenet.btxrd.subtype.selected",
            checkpoint_path=Path(fit_result["best_checkpoint"]),
            replace_alias=args.force_new_run,
            model_family="hosp_bonenet",
            phase="phase_b",
            source_run=run_dir,
            selected_metric={"name": "val_AUROC", "value": float(fit_result["best_AUROC"])},
            configuration_fingerprint=fingerprint,
            test_status={
                "status": "complete",
                "protocol": "btxrd_fresh_calibration_and_test",
                "test_evaluation_count": 1,
                "metrics_path": str(metrics_path.relative_to(root)),
                "predictions_sha256": _sha256(run_dir / "test_binary.csv"),
                "subtype_predictions_sha256": _sha256(run_dir / "test_subtype.csv"),
            },
        )
    _write_status(
        status_path,
        {
            "status": "complete",
            "configuration_fingerprint": fingerprint,
            "run_dir": str(run_dir),
            "metrics": str(metrics_path),
            "reusable": reusable,
        },
    )
    print(
        json.dumps(
            {"status": "complete", "run_dir": str(run_dir), "metrics": str(metrics_path)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
