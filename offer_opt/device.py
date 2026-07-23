"""The one place device selection happens. Every solver function takes a
`torch.device` and only ever calls `.to(device)` at tensor-construction
time -- scatter_reduce_/scatter_add_ have both CPU and CUDA kernels, so the
same code runs unmodified on CPU, CUDA, or Apple's MPS backend."""

from __future__ import annotations

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def dtype_for(device: torch.device) -> torch.dtype:
    """MPS (Apple GPU) has no float64 support; CPU/CUDA get full precision."""
    return torch.float32 if device.type == "mps" else torch.float64


def synchronize(device: torch.device) -> None:
    """Needed around benchmark timers -- CUDA/MPS ops are async."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
