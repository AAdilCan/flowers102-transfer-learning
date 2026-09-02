"""Tests for dataset/dataloader construction (against a faked Flowers102)."""

from __future__ import annotations

import pytest
import torch

from imgclf.data.class_names import CLASS_NAMES, NUM_CLASSES, class_name
from imgclf.data.dataset import build_cache_loader, build_dataloaders, build_dataset


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


def test_cache_loader_keeps_every_sample(base_config, fake_flowers):
    """The training loader drops a partial batch; the cache loader must not."""
    base_config.data.batch_size = 32  # 204 fake samples -> 6 full + 1 partial
    loader = build_cache_loader(base_config, "train", download=False)
    n_seen = sum(labels.numel() for _, labels in loader)
    assert n_seen == len(loader.dataset)
    assert n_seen % base_config.data.batch_size != 0  # a partial batch existed


def test_cache_loader_is_deterministic(base_config, fake_flowers):
    """Frozen features must be a stable function of the image, so no augmentation."""
    loader = build_cache_loader(base_config, "train", download=False)
    first = next(iter(loader))[0]
    second = next(iter(loader))[0]
    assert torch.equal(first, second)


def test_cache_loader_preserves_label_order(base_config, fake_flowers):
    loader = build_cache_loader(base_config, "train", download=False)
    labels = torch.cat([lbl for _, lbl in loader])
    assert torch.equal(labels[:5], torch.tensor([0, 1, 2, 3, 4]))


def test_train_loader_augmentation_can_be_disabled(base_config, fake_flowers):
    """augment_train=False swaps in the deterministic eval transform."""
    augmented = build_dataloaders(base_config, download=False, augment_train=True)
    plain = build_dataloaders(base_config, download=False, augment_train=False)
    sample_a = augmented.train.dataset[0][0]
    sample_b = plain.train.dataset[0][0]
    plain_again = plain.train.dataset[0][0]
    assert torch.equal(sample_b, plain_again)
    assert not torch.equal(sample_a, sample_b)
