"""Detect local compute capability.

PyTorch is optional. This module must keep working when it is not installed.
"""

from dataclasses import dataclass
from typing import Literal

PreferredDevice = Literal["mps", "cpu"]


@dataclass(frozen=True)
class ComputeCapability:
    preferred_device: PreferredDevice
    torch_installed: bool
    mps_available: bool


def _mps_is_available(torch_module: object) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def detect_compute() -> ComputeCapability:
    try:
        import torch
    except ImportError:
        return ComputeCapability(
            preferred_device="cpu",
            torch_installed=False,
            mps_available=False,
        )

    mps_available = _mps_is_available(torch)
    preferred_device: PreferredDevice = "mps" if mps_available else "cpu"
    return ComputeCapability(
        preferred_device=preferred_device,
        torch_installed=True,
        mps_available=mps_available,
    )
