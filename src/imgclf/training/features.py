"""Backbone feature caching for fast CPU linear probing.

When the backbone is frozen, its output for a given image never changes, so
recomputing it every epoch is pure waste — and on CPU it dominates runtime. We
run each split through the backbone exactly once, cache the pooled features, and
then train the linear head on those cached tensors for many cheap epochs.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..logging_utils import get_logger
from ..models import TransferModel

logger = get_logger("features")


@torch.no_grad()
def extract_features(
    model: TransferModel, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every batch through the backbone and stack pooled features + labels.

    Returns ``(features, labels)`` as CPU tensors of shape
    ``(N, feature_dim)`` and ``(N,)``.
    """
    model.eval()
    model.to(device)
    feats: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device)
        batch = model.extract_features(images)
        feats.append(batch.cpu())
        targets.append(labels.clone())
    features = torch.cat(feats)
    all_labels = torch.cat(targets)
    logger.info("cached %d feature vectors of dim %d", *tuple(features.shape))
    return features, all_labels


def build_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap cached ``(features, labels)`` tensors in a DataLoader."""
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
