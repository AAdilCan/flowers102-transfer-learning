"""Reproducibility and device helpers used across the pipeline."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for repeatable runs.

    This does not force fully deterministic CUDA kernels (which would cost
    throughput); it removes run-to-run variance from data shuffling and weight
    initialisation, which is what matters for comparing configurations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """Turn a device preference string into a concrete ``torch.device``.

    ``"auto"`` picks CUDA when available and falls back to CPU. Anything else
    is passed straight through so callers can force a device in tests.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters, optionally restricting to those that require grad."""
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad or not trainable_only
    )
