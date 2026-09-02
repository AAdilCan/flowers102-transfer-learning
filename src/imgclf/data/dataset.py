"""Dataset and DataLoader construction for Oxford Flowers-102.

Wraps ``torchvision.datasets.Flowers102`` so the rest of the codebase deals in
plain train/val/test loaders and never touches torchvision split naming or
transform wiring directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Flowers102

from ..config import Config
from ..logging_utils import get_logger
from .class_names import CLASS_NAMES, NUM_CLASSES
from .transforms import build_eval_transform, build_train_transform

logger = get_logger("data")

# torchvision uses these split identifiers; "val" is the official validation
# split, not a slice we carve out ourselves.
_TORCHVISION_SPLITS = {"train": "train", "val": "val", "test": "test"}


@dataclass
class DataBundle:
    """The three loaders plus the class-name lookup they share."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def build_dataset(
    cfg: Config, split: str, *, download: bool = True, augment: bool | None = None
) -> Dataset:
    """Instantiate one Flowers-102 split with the appropriate transform.

    By default the ``train`` split gets stochastic augmentation while ``val``
    and ``test`` get the deterministic eval pipeline. ``augment`` overrides that
    default, which the frozen-feature cache relies on: augmenting there would
    freeze one arbitrary random crop per image for the whole run.
    """
    if split not in _TORCHVISION_SPLITS:
        raise ValueError(
            f"split must be one of {sorted(_TORCHVISION_SPLITS)}, got {split!r}"
        )

    if augment is None:
        augment = split == "train"

    if augment:
        transform = build_train_transform(cfg.data.image_size, cfg.augmentation)
    else:
        transform = build_eval_transform(cfg.data.image_size)

    return Flowers102(
        root=cfg.data.root,
        split=_TORCHVISION_SPLITS[split],
        transform=transform,
        download=download,
    )


def build_dataloaders(
    cfg: Config, *, download: bool = True, augment_train: bool = True
) -> DataBundle:
    """Build train/val/test loaders from a :class:`Config`."""
    if cfg.model.num_classes != NUM_CLASSES:
        logger.warning(
            "config num_classes=%d but Flowers-102 has %d classes",
            cfg.model.num_classes,
            NUM_CLASSES,
        )

    train_ds = build_dataset(cfg, "train", download=download, augment=augment_train)
    val_ds = build_dataset(cfg, "val", download=download)
    test_ds = build_dataset(cfg, "test", download=download)

    common = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=cfg.data.drop_last, **common
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    logger.info(
        "loaders ready | train=%d val=%d test=%d | batch_size=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        cfg.data.batch_size,
    )
    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=CLASS_NAMES,
    )


def build_cache_loader(cfg: Config, split: str, *, download: bool = True) -> DataLoader:
    """Deterministic loader used to cache frozen-backbone features.

    Differs from the training loader in three ways that all matter for caching:
    no augmentation (the features must be a stable function of the image), no
    shuffling (features stay aligned with their labels in a readable order) and
    ``drop_last=False`` — the training loader drops the final partial batch for
    BatchNorm's sake, which would silently throw away up to ``batch_size - 1``
    of the 1,020 training images before the head ever sees them.
    """
    dataset = build_dataset(cfg, split, download=download, augment=False)
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False,
    )
