"""Explicit device selection and small synthetic compute checks; no model training."""

import torch


def get_device(request: str = "cuda") -> torch.device:
    """Honor CPU/CUDA requests and fail rather than silently downgrade unavailable CUDA."""
    device = torch.device(request)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {request}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; check driver and PyTorch build")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index unavailable: {index}")
        device = torch.device("cuda", index)
    return device


def verify_tensor_device(request: str) -> dict:
    """Run tensor-only matrix multiplication/backpropagation, using AMP on CUDA."""
    device = get_device(request)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    x = torch.ones((8, 8), device=device, requires_grad=True)
    with torch.autocast(device_type=device.type, enabled=is_cuda):
        loss = (x @ x).square().mean()
    loss.backward()
    finite = bool(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item())
    if not finite:
        raise RuntimeError(f"Non-finite tensor forward/backward result on {device}")
    if is_cuda:
        torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "amp": is_cuda,
        "finite_gradient": finite,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if is_cuda else 0,
    }
