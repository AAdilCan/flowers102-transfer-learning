"""Tests for the typed configuration layer."""

from __future__ import annotations

import pytest

from imgclf.config import Config


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.model.backbone == "resnet18"
    assert cfg.model.num_classes == 102
    assert cfg.optim.epochs > 0
    assert 0.0 <= cfg.augmentation.crop_scale_min <= 1.0


def test_yaml_round_trip(tmp_path):
    cfg = Config.from_dict({"seed": 7, "model": {"backbone": "resnet50"}})
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.seed == 7
    assert loaded.model.backbone == "resnet50"
    assert loaded.to_dict() == cfg.to_dict()


def test_partial_dict_fills_defaults():
    cfg = Config.from_dict({"optim": {"lr": 0.05}})
    assert cfg.optim.lr == 0.05
    assert cfg.optim.epochs == Config().optim.epochs


def test_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="backbone must be one of"):
        Config.from_dict({"model": {"backbone": "vgg16"}})


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        Config.from_dict({"model": {"mode": "frozen"}})


def test_rejects_unknown_top_level_key():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"learning_rate": 0.1})


def test_rejects_unknown_nested_key():
    with pytest.raises(ValueError, match="unknown keys for"):
        Config.from_dict({"data": {"batch": 8}})


@pytest.mark.parametrize(
    "section,payload,message",
    [
        ("data", {"image_size": 16}, "image_size"),
        ("data", {"batch_size": 0}, "batch_size"),
        ("data", {"num_workers": -1}, "num_workers"),
        ("augmentation", {"crop_scale_min": 0.0}, "crop_scale_min"),
        ("augmentation", {"crop_scale_min": 1.5}, "crop_scale_min"),
        ("augmentation", {"rotation_degrees": -5}, "rotation_degrees"),
        ("augmentation", {"color_jitter": -0.1}, "color_jitter"),
        ("model", {"num_classes": 1}, "num_classes"),
        ("model", {"dropout": 1.0}, "dropout"),
        ("model", {"backbone": "resnet101"}, "backbone"),
        ("model", {"mode": "frozen"}, "mode"),
        ("optim", {"epochs": 0}, "epochs"),
        ("optim", {"lr": 0.0}, "lr"),
        ("optim", {"weight_decay": -1e-4}, "weight_decay"),
        ("optim", {"min_lr_factor": 1.5}, "min_lr_factor"),
        ("optim", {"label_smoothing": 1.0}, "label_smoothing"),
        ("optim", {"early_stopping_patience": 0}, "early_stopping_patience"),
        ("optim", {"grad_clip_norm": 0.0}, "grad_clip_norm"),
    ],
)
def test_invalid_values_are_rejected(section, payload, message):
    with pytest.raises(ValueError, match=message):
        Config.from_dict({section: payload})


def test_valid_edge_values_are_accepted():
    cfg = Config.from_dict(
        {
            "data": {"image_size": 32, "batch_size": 1, "num_workers": 0},
            "augmentation": {"crop_scale_min": 1.0, "rotation_degrees": 0, "color_jitter": 0},
            "model": {"dropout": 0.0, "num_classes": 2},
            "optim": {"epochs": 1, "min_lr_factor": 0.0, "label_smoothing": 0.0,
                      "early_stopping_patience": 1, "grad_clip_norm": None},
        }
    )
    assert cfg.data.image_size == 32
    assert cfg.optim.grad_clip_norm is None
