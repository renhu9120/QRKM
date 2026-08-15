"""Shared helpers for dtype/device consistency."""

from __future__ import annotations

import torch


def assert_float64(*tensors: torch.Tensor, name: str = "tensor") -> None:
    for i, t in enumerate(tensors):
        if t.dtype != torch.float64:
            raise TypeError(f"{name}[{i}] must be torch.float64, got {t.dtype}")


def to_float64(t: torch.Tensor, *, name: str = "tensor") -> torch.Tensor:
    if t.dtype != torch.float64:
        return t.to(dtype=torch.float64)
    return t


def pick_device(preferred: str | torch.device) -> torch.device:
    if isinstance(preferred, torch.device):
        if preferred.type == "cuda" and torch.cuda.is_available():
            idx = 0 if preferred.index is None else int(preferred.index)
            return torch.device("cuda", idx)
        if preferred.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return preferred
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(preferred)
