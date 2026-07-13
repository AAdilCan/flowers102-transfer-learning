"""Tests for the backbone factory and the transfer model.

All backbones are built with ``pretrained=False`` so the suite never downloads
ImageNet weights; the wiring (feature dims, freezing, forward shapes) is what
matters here, not the specific weight values.
"""

from __future__ import annotations

import pytest
import torch

from imgclf.config import Config
from imgclf.models import build_backbone, build_model
from imgclf.models.classifier import TransferModel
from imgclf.utils import count_parameters

EXPECTED_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
}


def _config(backbone: str, mode: str = "linear_probe", num_classes: int = 10) -> Config:
    return Config.from_dict(
        {
            "device": "cpu",
            "model": {
                "backbone": backbone,
                "num_classes": num_classes,
                "pretrained": False,
                "mode": mode,
            },
        }
    )


@pytest.mark.parametrize("backbone,dim", EXPECTED_FEATURE_DIMS.items())
def test_backbone_feature_dim_and_shape(backbone: str, dim: int) -> None:
    extractor, feature_dim = build_backbone(backbone, pretrained=False)
    assert feature_dim == dim
    extractor.eval()
    with torch.no_grad():
        feats = extractor(torch.randn(2, 3, 64, 64))
    assert feats.shape == (2, dim)


def test_unknown_backbone_raises() -> None:
    with pytest.raises(ValueError, match="backbone must be one of"):
        build_backbone("vgg16", pretrained=False)


def test_invalid_mode_raises() -> None:
    _, feature_dim = build_backbone("resnet18", pretrained=False)
    extractor, _ = build_backbone("resnet18", pretrained=False)
    with pytest.raises(ValueError, match="mode must be"):
        TransferModel(extractor, feature_dim, 10, mode="bogus")


@pytest.mark.parametrize("backbone", list(EXPECTED_FEATURE_DIMS))
def test_forward_shape(backbone: str) -> None:
    model = build_model(_config(backbone), pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(3, 3, 64, 64))
    assert logits.shape == (3, 10)


def test_linear_probe_freezes_backbone() -> None:
    model = build_model(_config("resnet18", mode="linear_probe"), pretrained=False)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    # Only the head's two tensors (weight, bias) should be trainable.
    trainable = count_parameters(model, trainable_only=True)
    assert trainable == count_parameters(model.head, trainable_only=False)


def test_finetune_trains_everything() -> None:
    model = build_model(_config("resnet18", mode="finetune"), pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


def test_train_keeps_frozen_backbone_in_eval() -> None:
    model = build_model(_config("resnet18", mode="linear_probe"), pretrained=False)
    model.train()
    # Backbone must stay in eval so its BatchNorm stats are not updated.
    assert not model.backbone.training
    assert model.head.training


def test_cached_features_match_full_forward() -> None:
    model = build_model(_config("resnet18"), pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        direct = model(x)
        feats = model.extract_features(x)
        cached = model.classify_features(feats)
    assert feats.shape == (2, 512)
    torch.testing.assert_close(direct, cached)


def test_unfreeze_backbone() -> None:
    model = build_model(_config("resnet18", mode="linear_probe"), pretrained=False)
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
