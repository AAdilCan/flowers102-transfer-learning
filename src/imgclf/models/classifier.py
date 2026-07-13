"""Classifier head and the full transfer-learning model.

:class:`TransferModel` stacks a small dropout + linear head on top of a headless
backbone (see :mod:`imgclf.models.backbones`) and encodes the two training
regimes this project compares:

``linear_probe``
    The backbone is frozen and kept in eval mode, so only the head trains and
    the (expensive) features can be precomputed once and cached.

``finetune``
    The whole network is trainable and updated end-to-end.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from ..config import Config
from .backbones import build_backbone


class ClassifierHead(nn.Module):
    """Dropout followed by a single linear layer mapping features to logits."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(features))


class TransferModel(nn.Module):
    """A headless backbone plus a classifier head, wired for one training mode."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.2,
        mode: str = "linear_probe",
    ):
        super().__init__()
        if mode not in ("linear_probe", "finetune"):
            raise ValueError(
                f"mode must be 'linear_probe' or 'finetune', got {mode!r}"
            )
        self.backbone = backbone
        self.head = ClassifierHead(feature_dim, num_classes, dropout)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.mode = mode

        if mode == "linear_probe":
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Disable gradients on the backbone and pin it to eval mode.

        Eval mode matters as much as ``requires_grad``: it stops BatchNorm from
        updating its running statistics and disables backbone dropout, so the
        cached features are stable across epochs.
        """
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients on the backbone for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features ``(batch, feature_dim)``."""
        return self.backbone(images)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Map precomputed features to logits — the cached linear-probe path."""
        return self.head(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the parameters the optimiser should update."""
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True) -> "TransferModel":
        """Switch to train mode, keeping a frozen backbone in eval.

        The default ``nn.Module.train`` would flip the whole tree — including a
        linear-probe backbone — back into training mode, silently un-freezing
        its BatchNorm layers. Overriding it keeps the freeze intact.
        """
        super().train(mode)
        if self.mode == "linear_probe":
            self.backbone.eval()
        return self


def build_model(cfg: Config, *, pretrained: bool | None = None) -> TransferModel:
    """Construct a :class:`TransferModel` from a :class:`Config`.

    ``pretrained`` overrides ``cfg.model.pretrained`` when given, which lets the
    test suite build a randomly-initialised network without downloading weights.
    """
    use_pretrained = cfg.model.pretrained if pretrained is None else pretrained
    backbone, feature_dim = build_backbone(
        cfg.model.backbone, pretrained=use_pretrained
    )
    return TransferModel(
        backbone,
        feature_dim,
        cfg.model.num_classes,
        dropout=cfg.model.dropout,
        mode=cfg.model.mode,
    )
