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
