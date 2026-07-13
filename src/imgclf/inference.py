"""Inference helpers: rebuild a model from a checkpoint, score a loader, predict.

These back the ``evaluate`` and ``predict`` CLI commands. The richer evaluation
harness (per-class metrics, confusion matrix, curves) builds on
:func:`collect_predictions` and lives in :mod:`imgclf.eval`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import Config
from .data.class_names import CLASS_NAMES
from .data.transforms import build_eval_transform
from .logging_utils import get_logger
from .models import TransferModel, build_model
from .training.checkpoint import load_checkpoint

logger = get_logger("inference")


@dataclass
class Prediction:
    """A single top-k prediction: parallel label / name / probability lists."""

    labels: list[int]
    names: list[str]
    probs: list[float]


def load_model_from_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[TransferModel, Config]:
    """Rebuild the model described by a checkpoint and load its weights."""
    payload = load_checkpoint(path, map_location=str(device))
    cfg = Config.from_dict(payload["config"])
    # Skip weight download: the checkpoint already carries trained parameters.
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    logger.info("loaded model from %s (epoch %s)", path, payload.get("epoch"))
    return model, cfg


@torch.no_grad()
def collect_predictions(
    model: TransferModel, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked ``(logits, labels)`` over a loader for offline metrics."""
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for images, labels in loader:
        logits = model(images.to(device))
        logits_all.append(logits.cpu())
        labels_all.append(labels.clone())
    return torch.cat(logits_all), torch.cat(labels_all)


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """Fraction of samples whose true label is within the top-``k`` logits."""
    topk = logits.topk(k, dim=1).indices
    hits = (topk == labels.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


@torch.no_grad()
def predict_image(
    model: TransferModel,
    image_path: str | Path,
    device: torch.device,
    *,
    image_size: int = 224,
    top_k: int = 5,
) -> Prediction:
    """Classify one image file and return its top-``k`` predictions."""
    model.eval()
    transform = build_eval_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probs = model(tensor).softmax(dim=1).squeeze(0)
    top = probs.topk(min(top_k, probs.numel()))
    labels = top.indices.tolist()
    return Prediction(
        labels=labels,
        names=[CLASS_NAMES[i] for i in labels],
        probs=top.values.tolist(),
    )
