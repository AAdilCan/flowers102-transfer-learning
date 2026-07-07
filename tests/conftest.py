"""Shared pytest fixtures.

Tests must run without the ~330 MB Flowers-102 download, so the dataset is
faked at the torchvision boundary. Everything above that boundary (transforms,
loader wiring, config) is exercised against the fake.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from imgclf.config import Config


@pytest.fixture()
def base_config(tmp_path) -> Config:
    """A small, fast config pointing data/output at a temp dir."""
    return Config.from_dict(
        {
            "seed": 0,
            "device": "cpu",
            "output_dir": str(tmp_path / "ckpt"),
            "data": {
                "root": str(tmp_path / "data"),
                "image_size": 64,
                "batch_size": 4,
                "num_workers": 0,
            },
            "optim": {"epochs": 1},
        }
    )


class FakeFlowers:
    """Drop-in stand-in for torchvision's Flowers102.

    Generates random RGB images of varying size on the fly and applies whatever
    transform it was constructed with, mirroring the real dataset's interface
    (``__len__``, ``__getitem__``, ``_labels``).
    """

    def __init__(self, root, split, transform=None, download=False, n_per_class=2, n_classes=102):
        self.root = root
        self.split = split
        self.transform = transform
        self._n = n_per_class * n_classes
        self._labels = [i % n_classes for i in range(self._n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        label = self._labels[idx]
        # Vary the size so resize/crop transforms are genuinely exercised.
        size = 80 + (idx % 5) * 10
        img = Image.new("RGB", (size, size + 8), color=(idx % 255, 100, 150))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


@pytest.fixture()
def fake_flowers(monkeypatch):
    """Patch the Flowers102 symbol used inside the dataset module."""
    import imgclf.data.dataset as dataset_mod

    monkeypatch.setattr(dataset_mod, "Flowers102", FakeFlowers)
    return FakeFlowers
