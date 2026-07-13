"""End-to-end CLI tests driven through click's in-process test runner.

A single tiny YAML config (random backbone, faked dataset) is trained once, then
reused by the evaluate and predict commands so the suite pays the training cost
only once.
"""

from __future__ import annotations

import yaml
from click.testing import CliRunner
from PIL import Image

from imgclf.cli import cli


def _write_config(path, tmp_path) -> None:
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "ckpt"),
        "data": {
            "root": str(tmp_path / "data"),
            "image_size": 64,
            "batch_size": 8,
            "num_workers": 0,
        },
        "model": {"backbone": "resnet18", "pretrained": False, "mode": "linear_probe"},
        "optim": {"epochs": 1, "lr": 1e-2},
    }
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)


def test_cli_train_evaluate_predict(tmp_path, fake_flowers) -> None:
    runner = CliRunner()
    config_path = tmp_path / "cfg.yaml"
    _write_config(config_path, tmp_path)

    train_result = runner.invoke(
        cli, ["train", "--config", str(config_path), "--no-download"]
    )
    assert train_result.exit_code == 0, train_result.output
    ckpt = tmp_path / "ckpt" / "best.pt"
    assert ckpt.exists()
    assert "best val acc" in train_result.output

    eval_result = runner.invoke(
        cli, ["evaluate", "--checkpoint", str(ckpt), "--split", "val", "--no-download"]
    )
    assert eval_result.exit_code == 0, eval_result.output
    assert "top-1" in eval_result.output and "top-5" in eval_result.output

    img_path = tmp_path / "q.png"
    Image.new("RGB", (90, 90), color=(10, 180, 60)).save(img_path)
    predict_result = runner.invoke(
        cli, ["predict", "--checkpoint", str(ckpt), "--image", str(img_path), "--top-k", "3"]
    )
    assert predict_result.exit_code == 0, predict_result.output
    assert len(predict_result.output.strip().splitlines()) == 3


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "evaluate", "predict"):
        assert command in result.output
