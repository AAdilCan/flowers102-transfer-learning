"""Checkpoint save/load.

A checkpoint bundles the model weights with the config that produced them and
the metrics at save time, so a checkpoint is self-describing: ``evaluate`` and
``predict`` can rebuild the exact architecture without being told the backbone
or mode on the command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import Config
from ..logging_utils import get_logger

logger = get_logger("checkpoint")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: Config,
    *,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Write model state, config and metrics to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)
    logger.info("saved checkpoint -> %s (epoch %d)", path, epoch)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload from disk.

    Returns the raw dict; callers rebuild the model from ``payload['config']``
    and load ``payload['model_state']`` into it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
