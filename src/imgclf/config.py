"""Typed configuration for the training/evaluation pipeline.

The whole run is driven by a single :class:`Config` object. It carries nested
dataclasses for data, model, augmentation and optimisation so that every knob
lives in one place and can be round-tripped to YAML for reproducibility.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# ImageNet channel statistics — the pretrained backbones were trained with
# these, so inputs must be normalised the same way at train and inference time.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")


@dataclass
class DataConfig:
    """Where the dataset lives and how loaders are built."""

    root: str = "data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    # Drop the last incomplete batch during training so BatchNorm statistics
    # are never estimated from a single-sample batch.
    drop_last: bool = True

    def __post_init__(self) -> None:
        # Catching these here beats a shape error thrown deep inside a
        # DataLoader worker twenty minutes into a run.
        if self.image_size < 32:
            raise ValueError(f"image_size must be >= 32, got {self.image_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")


@dataclass
class AugmentationConfig:
    """Train-time augmentation. Eval uses a deterministic resize + centre crop."""

    horizontal_flip: bool = True
    rotation_degrees: float = 20.0
    color_jitter: float = 0.2
    random_resized_crop: bool = True
    # Lower bound of the area fraction kept by RandomResizedCrop.
    crop_scale_min: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError(
                f"crop_scale_min must be in (0, 1], got {self.crop_scale_min}"
            )
        if self.rotation_degrees < 0:
            raise ValueError(f"rotation_degrees must be >= 0, got {self.rotation_degrees}")
        if self.color_jitter < 0:
            raise ValueError(f"color_jitter must be >= 0, got {self.color_jitter}")


@dataclass
class ModelConfig:
    """Backbone choice and how much of it is trainable."""

    backbone: str = "resnet18"
    num_classes: int = 102
    pretrained: bool = True
    dropout: float = 0.2
    # "linear_probe" freezes the backbone and trains only the head;
    # "finetune" unfreezes the whole network.
    mode: str = "linear_probe"

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONES}, got {self.backbone!r}"
            )
        if self.mode not in ("linear_probe", "finetune"):
            raise ValueError(
                f"mode must be 'linear_probe' or 'finetune', got {self.mode!r}"
            )
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class OptimConfig:
    """Optimiser, schedule and regularisation."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Cosine annealing warms down to this fraction of the base LR.
    min_lr_factor: float = 0.01
    label_smoothing: float = 0.1
    early_stopping_patience: int = 7
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not 0.0 <= self.min_lr_factor <= 1.0:
            raise ValueError(f"min_lr_factor must be in [0, 1], got {self.min_lr_factor}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}"
            )
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be > 0, got {self.grad_clip_norm}")


@dataclass
class Config:
    """Top-level run configuration."""

    seed: int = 42
    device: str = "auto"  # "auto" -> cuda if available else cpu
    output_dir: str = "checkpoints"
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Serialise the config to YAML for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """Build a Config from a (possibly partial) nested dict."""
        raw = dict(raw or {})
        nested = {
            "data": DataConfig,
            "augmentation": AugmentationConfig,
            "model": ModelConfig,
            "optim": OptimConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in nested.items():
            section = raw.pop(name, None)
            kwargs[name] = _build(klass, section)
        # Whatever is left are the scalar top-level fields.
        top_fields = {f.name for f in dataclasses.fields(cls)}
        for key, value in raw.items():
            if key not in top_fields:
                raise ValueError(f"unknown config key: {key!r}")
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)


def _build(klass: type, section: dict[str, Any] | None) -> Any:
    """Instantiate a nested dataclass, validating unknown keys."""
    if section is None:
        return klass()
    allowed = {f.name for f in dataclasses.fields(klass)}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown keys for {klass.__name__}: {sorted(unknown)}")
    return klass(**section)
