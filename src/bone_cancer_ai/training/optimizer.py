"""Fit-only class weighting and the common AdamW optimizer."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
import torch
from torch import nn


def class_weight_from_frame(
    frame: pd.DataFrame, label_column: str = "binary_label"
) -> torch.Tensor:
    """Return ``n_negative / n_positive`` from an eligible fit manifest."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required_columns = {label_column, "assigned_split", "is_eligible"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"fit frame is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("fit frame must not be empty")
    if not frame["assigned_split"].eq("fit_train").all():
        raise ValueError("class weights may use only fit_train rows")
    if not frame["is_eligible"].eq(True).all():
        raise ValueError("class weights require eligible fit rows")

    labels = frame[label_column]
    if labels.isna().any() or not pd.api.types.is_numeric_dtype(labels.dtype):
        raise ValueError("fit binary labels must be present numeric values")
    numeric = labels.astype(float)
    if not numeric.isin([0.0, 1.0]).all():
        raise ValueError("fit binary labels must contain only 0 and 1")
    negatives = int((numeric == 0.0).sum())
    positives = int((numeric == 1.0).sum())
    if negatives == 0 or positives == 0:
        raise ValueError("fit binary labels must contain both classes")
    return torch.tensor(negatives / positives, dtype=torch.float32)


def build_optimizer(
    model_or_parameters: nn.Module | Iterable[nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build the single AdamW path used by every approved baseline."""

    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    source = (
        model_or_parameters.parameters()
        if isinstance(model_or_parameters, nn.Module)
        else model_or_parameters
    )
    parameters = [parameter for parameter in source if parameter.requires_grad]
    if not parameters:
        raise ValueError("optimizer requires at least one trainable parameter")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
