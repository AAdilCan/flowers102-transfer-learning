"""Tests for the training pipeline, checkpointing and inference helpers.

Everything runs on the faked Flowers-102 with randomly-initialised backbones, so
these assert on *wiring and shapes* (a checkpoint appears, predictions have the
right form, metrics are in range) rather than on learned accuracy.
"""

from __future__ import annotations

import torch
from PIL import Image

from imgclf.config import Config
from imgclf.inference import (
    collect_predictions,
    load_model_from_checkpoint,
    predict_image,
    top_k_accuracy,
)
from imgclf.models import build_model
from imgclf.training import fit, load_checkpoint, save_checkpoint, train_from_config


def _tiny_config(tmp_path, mode: str = "linear_probe", epochs: int = 2) -> Config:
    return Config.from_dict(
        {
            "seed": 0,
            "device": "cpu",
            "output_dir": str(tmp_path / "ckpt"),
            "data": {
                "root": str(tmp_path / "data"),
                "image_size": 64,
                "batch_size": 8,
                "num_workers": 0,
            },
            "model": {"backbone": "resnet18", "pretrained": False, "mode": mode},
            "optim": {"epochs": epochs, "lr": 1e-2, "early_stopping_patience": 10},
        }
    )


def test_pipeline_linear_probe_and_inference(tmp_path, fake_flowers) -> None:
    cfg = _tiny_config(tmp_path, mode="linear_probe", epochs=2)
    history = train_from_config(cfg, download=False)

    assert 0.0 <= history.best_val_acc <= 1.0
    assert history.best_epoch >= 1
    assert len(history.train_loss) == len(history.val_loss)

    ckpt = tmp_path / "ckpt" / "best.pt"
    assert ckpt.exists()
    assert (tmp_path / "ckpt" / "config.yaml").exists()

    # Round-trip through the checkpoint and score the val split.
    device = torch.device("cpu")
    model, loaded_cfg = load_model_from_checkpoint(ckpt, device)
    assert loaded_cfg.model.backbone == "resnet18"

    from imgclf.data import build_dataloaders

    bundle = build_dataloaders(loaded_cfg, download=False)
    logits, labels = collect_predictions(model, bundle.val, device)
    assert logits.shape[0] == labels.shape[0]
    top1 = top_k_accuracy(logits, labels, k=1)
    top5 = top_k_accuracy(logits, labels, k=5)
    assert 0.0 <= top1 <= top5 <= 1.0


def test_pipeline_finetune_runs(tmp_path, fake_flowers) -> None:
    cfg = _tiny_config(tmp_path, mode="finetune", epochs=1)
    history = train_from_config(cfg, download=False)
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert len(history.train_acc) == 1


def test_early_stopping_triggers(tmp_path) -> None:
    # Constant labels + tiny random features: val accuracy plateaus fast, so
    # early stopping should cut training well before the epoch budget.
    cfg = _tiny_config(tmp_path, mode="linear_probe", epochs=50)
    cfg.optim.early_stopping_patience = 2
    cfg.model.num_classes = 3
    model = build_model(cfg, pretrained=False)

    feats = torch.randn(24, model.feature_dim)
    labels = torch.zeros(24, dtype=torch.long)
    from imgclf.training import build_feature_loader

    loader = build_feature_loader(feats, labels, batch_size=8, shuffle=False)
    history = fit(model, cfg, loader, loader, torch.device("cpu"), on_features=True)
    assert len(history.train_loss) < cfg.optim.epochs


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = _tiny_config(tmp_path)
    model = build_model(cfg, pretrained=False)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, cfg, epoch=3, metrics={"val_acc": 0.5})

    payload = load_checkpoint(path)
    assert payload["epoch"] == 3
    assert payload["metrics"]["val_acc"] == 0.5

    reloaded = build_model(Config.from_dict(payload["config"]), pretrained=False)
    reloaded.load_state_dict(payload["model_state"])
    a = next(model.head.parameters())
    b = next(reloaded.head.parameters())
    torch.testing.assert_close(a, b)


def test_top_k_accuracy_exact() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    labels = torch.tensor([1, 0])
    assert top_k_accuracy(logits, labels, k=1) == 1.0
    wrong = torch.tensor([0, 1])
    assert top_k_accuracy(logits, wrong, k=1) == 0.0
    assert top_k_accuracy(logits, wrong, k=3) == 1.0


def test_predict_image(tmp_path, fake_flowers) -> None:
    cfg = _tiny_config(tmp_path, epochs=1)
    train_from_config(cfg, download=False)
    ckpt = tmp_path / "ckpt" / "best.pt"

    model, loaded_cfg = load_model_from_checkpoint(ckpt, torch.device("cpu"))
    img_path = tmp_path / "flower.png"
    Image.new("RGB", (120, 100), color=(120, 200, 90)).save(img_path)

    result = predict_image(
        model, img_path, torch.device("cpu"),
        image_size=loaded_cfg.data.image_size, top_k=3,
    )
    assert len(result.names) == 3 == len(result.probs) == len(result.labels)
    assert all(0.0 <= p <= 1.0 for p in result.probs)
