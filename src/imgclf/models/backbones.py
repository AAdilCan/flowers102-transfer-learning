"""Pretrained backbone factory.

Each backbone is loaded from torchvision with its final ImageNet classifier
stripped to a :class:`~torch.nn.Identity`, so calling it returns the pooled
feature vector ``(batch, feature_dim)`` instead of 1000-way logits. The head in
:mod:`imgclf.models.classifier` is stacked on top of that vector, which keeps
the "extract features once, classify many times" path (used for cached linear
probing) trivial: the backbone *is* the feature extractor.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torchvision import models

from ..config import SUPPORTED_BACKBONES
from ..logging_utils import get_logger

logger = get_logger("models")


def _resnet_builder(
    ctor: Callable[..., nn.Module], weights_enum: type
) -> Callable[[bool], tuple[nn.Module, int]]:
    """Return a builder that strips a ResNet's ``fc`` layer."""

    def build(pretrained: bool) -> tuple[nn.Module, int]:
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        net = ctor(weights=weights)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim

    return build


def _efficientnet_builder(
    ctor: Callable[..., nn.Module], weights_enum: type
) -> Callable[[bool], tuple[nn.Module, int]]:
    """Return a builder that strips an EfficientNet's ``classifier`` block."""

    def build(pretrained: bool) -> tuple[nn.Module, int]:
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        net = ctor(weights=weights)
        # classifier is Sequential(Dropout, Linear); the Linear holds the dim.
        feature_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        return net, feature_dim

    return build


# Registry mapping the supported names to a builder returning
# ``(feature_extractor, feature_dim)``.
_BUILDERS: dict[str, Callable[[bool], tuple[nn.Module, int]]] = {
    "resnet18": _resnet_builder(models.resnet18, models.ResNet18_Weights),
    "resnet50": _resnet_builder(models.resnet50, models.ResNet50_Weights),
    "efficientnet_b0": _efficientnet_builder(
        models.efficientnet_b0, models.EfficientNet_B0_Weights
    ),
}


def build_backbone(name: str, *, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a headless pretrained backbone and report its feature dimension.

    Parameters
    ----------
    name:
        One of :data:`imgclf.config.SUPPORTED_BACKBONES`.
    pretrained:
        Load ImageNet weights when ``True``; random init otherwise (used by the
        test suite so it never needs to download weights).

    Returns
    -------
    tuple[nn.Module, int]
        The feature extractor whose forward yields ``(batch, feature_dim)`` and
        the ``feature_dim`` itself.
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"backbone must be one of {SUPPORTED_BACKBONES}, got {name!r}"
        )
    backbone, feature_dim = _BUILDERS[name](pretrained)
    logger.info(
        "built backbone %s | feature_dim=%d | pretrained=%s",
        name,
        feature_dim,
        pretrained,
    )
    return backbone, feature_dim
