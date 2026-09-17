"""Grad-CAM for spatial layers in binary classifiers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class GradCAM:
    """Capture one layer and generate normalized class-specific activation maps."""

    def __init__(
        self, model: nn.Module, target_layer: nn.Module, *, output_key: str = "binary_logits"
    ) -> None:
        self.model = model
        self.output_key = output_key
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._capture_activation)

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _capture_activation(
        self, module: nn.Module, inputs: tuple[object, ...], output: object
    ) -> None:
        del module, inputs
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target layer must return a BCHW tensor")
        self._activation = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self._gradient = gradient

    def generate(
        self, images: torch.Tensor, *, target_classes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return one [0,1] map per image, resized to the input dimensions."""
        if not self.active:
            raise RuntimeError("Grad-CAM hooks are closed")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        states = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            output = self.model(images)
            if not isinstance(output, Mapping) or self.output_key not in output:
                raise ValueError(f"model output must contain {self.output_key}")
            logits = output[self.output_key]
            if not isinstance(logits, torch.Tensor) or logits.numel() != images.shape[0]:
                raise ValueError("binary logits must contain one value per image")
            logits = logits.reshape(images.shape[0])
            if target_classes is None:
                target_classes = (logits.detach() >= 0).to(torch.long)
            target_classes = torch.as_tensor(target_classes, device=logits.device).reshape(-1)
            if target_classes.shape != logits.shape or not bool(
                ((target_classes == 0) | (target_classes == 1)).all()
            ):
                raise ValueError("target_classes must contain one binary class per image")
            signs = target_classes.to(logits.dtype).mul(2.0).sub(1.0)
            torch.sum(logits * signs).backward()
            if self._activation is None or self._gradient is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
            weights = self._gradient.mean(dim=(-2, -1), keepdim=True)
            maps = torch.relu(torch.sum(weights * self._activation, dim=1, keepdim=True))
            maps = F.interpolate(
                maps, size=images.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
            flat = maps.flatten(1)
            minima = flat.min(dim=1).values[:, None, None]
            maxima = flat.max(dim=1).values[:, None, None]
            maps = (maps - minima) / (maxima - minima).clamp_min(1e-12)
            return maps.detach()
        finally:
            for module, training in states:
                module.training = training

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._activation = None
        self._gradient = None

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["GradCAM"]
