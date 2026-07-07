"""Tests for dataset/dataloader construction (against a faked Flowers102)."""

from __future__ import annotations

import pytest
import torch

from imgclf.data.class_names import CLASS_NAMES, NUM_CLASSES, class_name
from imgclf.data.dataset import build_dataloaders, build_dataset


def test_class_names_cover_all_labels():
    assert NUM_CLASSES == 102
    assert len(set(CLASS_NAMES)) == 102  # no duplicates
    assert class_name(0) == "pink primrose"
    assert class_name(101) == "blackberry lily"


def test_class_name_out_of_range():
    with pytest.raises(IndexError):
        class_name(102)


def test_build_dataset_applies_transform(base_config, fake_flowers):
    ds = build_dataset(base_config, "train", download=False)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)
    assert isinstance(label, int)


def test_build_dataset_rejects_bad_split(base_config, fake_flowers):
    with pytest.raises(ValueError, match="split must be one of"):
        build_dataset(base_config, "holdout", download=False)


def test_build_dataloaders_shapes(base_config, fake_flowers):
    bundle = build_dataloaders(base_config, download=False)
    assert bundle.num_classes == 102
    images, labels = next(iter(bundle.train))
    assert images.shape == (base_config.data.batch_size, 3, 64, 64)
    assert labels.shape == (base_config.data.batch_size,)


def test_train_loader_shuffles_val_does_not(base_config, fake_flowers):
    bundle = build_dataloaders(base_config, download=False)
    assert bundle.train.sampler is not None
    # Val/test iterate in dataset order (sequential sampler).
    val_labels = torch.cat([lbl for _, lbl in bundle.val])
    assert torch.equal(val_labels[:5], torch.tensor([0, 1, 2, 3, 4]))
