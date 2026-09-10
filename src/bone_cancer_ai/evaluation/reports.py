"""Report formatting, atomic persistence, and inference timing."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch

BASELINE_SUMMARY_COLUMNS = [
    "model",
    "seed",
    "best_epoch",
    "AUROC",
    "AUPRC",
    "balanced_accuracy",
    "F1",
    "sensitivity",
    "specificity",
    "MCC",
    "params",
    "latency_ms",
]


def confusion_matrix_frame(metrics: Mapping[str, object]) -> pd.DataFrame:
    """Format a ``[[TN, FP], [FN, TP]]`` matrix with explicit axis labels."""
    if "confusion_matrix" not in metrics:
        raise ValueError("metrics must contain a confusion_matrix")
    frame = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    )
    if frame.shape != (2, 2):
        raise ValueError("confusion_matrix must have shape 2x2")
    return frame


def _temporary_sibling(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")


def atomic_write_json(payload: object, destination: Path) -> None:
    """Serialize strict JSON to a sibling temporary file before replacement."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Serialize a table to a sibling temporary file before replacement."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def update_baseline_summary(destination: Path, row: Mapping[str, object]) -> None:
    """Atomically upsert one executed baseline result by model and seed."""
    if set(row) != set(BASELINE_SUMMARY_COLUMNS):
        missing = sorted(set(BASELINE_SUMMARY_COLUMNS) - set(row))
        extra = sorted(set(row) - set(BASELINE_SUMMARY_COLUMNS))
        raise ValueError(f"baseline summary row schema mismatch; missing={missing}, extra={extra}")
    destination = Path(destination)
    incoming = pd.DataFrame(
        [[row[column] for column in BASELINE_SUMMARY_COLUMNS]],
        columns=BASELINE_SUMMARY_COLUMNS,
    )
    if destination.exists():
        existing = pd.read_csv(destination)
        if existing.columns.tolist() != BASELINE_SUMMARY_COLUMNS:
            raise ValueError("existing baseline summary has an invalid schema")
        keep = ~(
            existing["model"].eq(row["model"])
            & existing["seed"].eq(row["seed"])
        )
        incoming = pd.concat([existing.loc[keep], incoming], ignore_index=True)
    incoming = incoming.sort_values(["model", "seed"], kind="stable").reset_index(drop=True)
    atomic_write_csv(incoming, destination)


def measure_inference_latency(
    model: torch.nn.Module,
    example: torch.Tensor,
    device: str | torch.device,
    warmup: int = 10,
    repetitions: int = 50,
) -> dict[str, object]:
    """Measure mean single-forward latency while preserving the model's mode."""
    if (
        type(warmup) is not int
        or type(repetitions) is not int
        or warmup <= 0
        or repetitions <= 0
    ):
        raise ValueError("warmup and repetitions must be positive integers")
    resolved_device = torch.device(device)
    if not isinstance(example, torch.Tensor):
        raise TypeError("example must be a torch.Tensor")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA latency requested but CUDA is unavailable")

    example = example.to(resolved_device)
    module_training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                model(example)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
            started = time.perf_counter()
            for _ in range(repetitions):
                model(example)
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
            elapsed_ms = (time.perf_counter() - started) * 1_000.0
    finally:
        for module, training in module_training_states:
            module.training = training

    return {
        "device": str(resolved_device),
        "warmup": warmup,
        "repetitions": repetitions,
        "total_elapsed_ms": float(elapsed_ms),
        "latency_ms": float(elapsed_ms / repetitions),
    }
