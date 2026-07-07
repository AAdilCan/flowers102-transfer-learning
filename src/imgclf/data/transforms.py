"""Image transform pipelines for training and evaluation.

Training transforms are stochastic (crop, flip, rotation, jitter) to expand the
tiny 10-images-per-class training set. Evaluation transforms are deterministic
so that metrics are reproducible: a fixed resize to a slightly larger edge
followed by a centre crop to the model input size.
"""

from __future__ import annotations

from torchvision import transforms

from ..config import AugmentationConfig, IMAGENET_MEAN, IMAGENET_STD

# Fraction by which eval images are resized before the centre crop, e.g. a
# 224px crop is taken from a 256px resize (224 / 0.875 ≈ 256).
_EVAL_RESIZE_RATIO = 0.875


def build_train_transform(
    image_size: int, aug: AugmentationConfig
) -> transforms.Compose:
    """Compose the stochastic training-time augmentation pipeline."""
    ops: list[object] = []

    if aug.random_resized_crop:
        ops.append(
            transforms.RandomResizedCrop(
                image_size, scale=(aug.crop_scale_min, 1.0)
            )
        )
    else:
        ops.append(transforms.Resize(_resize_edge(image_size)))
        ops.append(transforms.CenterCrop(image_size))

    if aug.horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())

    if aug.rotation_degrees > 0:
        ops.append(transforms.RandomRotation(aug.rotation_degrees))

    if aug.color_jitter > 0:
        j = aug.color_jitter
        ops.append(
            transforms.ColorJitter(brightness=j, contrast=j, saturation=j)
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Compose the deterministic evaluation pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(_resize_edge(image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _resize_edge(image_size: int) -> int:
    """Shorter-edge resize target that leaves margin for the centre crop."""
    return int(round(image_size / _EVAL_RESIZE_RATIO))
