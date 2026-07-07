"""Tests for the augmentation / eval transform pipelines."""

from __future__ import annotations

import torch
from PIL import Image

from imgclf.config import AugmentationConfig
from imgclf.data.transforms import build_eval_transform, build_train_transform


def _img(w=100, h=120):
    return Image.new("RGB", (w, h), color=(120, 60, 200))


def test_eval_transform_output_shape_and_type():
    t = build_eval_transform(64)
    out = t(_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32


def test_eval_transform_is_deterministic():
    t = build_eval_transform(64)
    img = _img()
    a = t(img)
    b = t(img)
    assert torch.equal(a, b)


def test_train_transform_output_shape():
    t = build_train_transform(64, AugmentationConfig())
    out = t(_img())
    assert out.shape == (3, 64, 64)


def test_train_transform_is_stochastic():
    # With augmentation enabled, two passes of the same image should differ.
    torch.manual_seed(0)
    t = build_train_transform(64, AugmentationConfig())
    img = _img()
    a = t(img)
    b = t(img)
    assert not torch.equal(a, b)


def test_normalization_shifts_range():
    # ImageNet normalisation pushes values outside [0, 1].
    t = build_eval_transform(64)
    out = t(_img())
    assert out.min() < 0.0


def test_augmentation_can_be_disabled():
    aug = AugmentationConfig(
        horizontal_flip=False,
        rotation_degrees=0.0,
        color_jitter=0.0,
        random_resized_crop=False,
    )
    t = build_train_transform(64, aug)
    img = _img()
    # Only deterministic ops remain -> repeatable output.
    assert torch.equal(t(img), t(img))
