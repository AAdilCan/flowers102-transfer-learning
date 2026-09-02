"""Command-line interface: ``imgclf train | evaluate | predict``.

The CLI is deliberately thin — it parses flags into a :class:`Config`, resolves
a device, and delegates to the training/inference modules. A base config can be
supplied as YAML and selectively overridden by flags, so experiments are both
reproducible (the YAML) and quick to tweak (the flags).
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .config import SUPPORTED_BACKBONES, Config
from .data import build_dataloaders
from .eval import build_report
from .data.class_names import class_name
from .inference import (
    collect_predictions,
    load_image_tensor,
    load_model_from_checkpoint,
    predict_image,
)
from .interpret import explain as gradcam_explain
from .interpret import save_gradcam_panel
from .reporting import write_figures
from .logging_utils import configure_logging
from .training import train_from_config
from .utils import resolve_device


def _load_base_config(config_path: str | None) -> Config:
    return Config.load(config_path) if config_path else Config()


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Transfer-learning image classifier for Oxford Flowers-102."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Base YAML config; flags below override its fields.")
@click.option("--backbone", type=click.Choice(SUPPORTED_BACKBONES), default=None)
@click.option("--mode", type=click.Choice(["linear_probe", "finetune"]), default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--output-dir", type=click.Path(), default=None)
@click.option("--data-root", type=click.Path(), default=None)
@click.option("--no-download", is_flag=True, help="Fail instead of downloading the dataset.")
def train(config_path, backbone, mode, epochs, lr, batch_size, output_dir, data_root, no_download):
    """Train a model and write the best checkpoint to the output dir."""
    cfg = _load_base_config(config_path)
    if backbone is not None:
        cfg.model.backbone = backbone
    if mode is not None:
        cfg.model.mode = mode
    if epochs is not None:
        cfg.optim.epochs = epochs
    if lr is not None:
        cfg.optim.lr = lr
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    if output_dir is not None:
        cfg.output_dir = output_dir
    if data_root is not None:
        cfg.data.root = data_root

    history = train_from_config(cfg, download=not no_download)
    click.echo(
        f"best val acc {history.best_val_acc:.4f} at epoch {history.best_epoch} "
        f"-> {Path(cfg.output_dir) / 'best.pt'}"
    )


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--split", type=click.Choice(["val", "test"]), default="test")
@click.option("--report-dir", type=click.Path(), default=None,
              help="Write metrics JSON, per-class CSV and figures here.")
@click.option("--history", "history_path", type=click.Path(exists=True), default=None,
              help="history.json from training; adds the training-curve figure.")
@click.option("--no-figures", is_flag=True, help="Skip plotting, write metrics only.")
@click.option("--no-download", is_flag=True)
def evaluate(checkpoint, split, report_dir, history_path, no_figures, no_download):
    """Score a checkpoint on a split and optionally write the full report."""
    device = resolve_device("auto")
    model, cfg = load_model_from_checkpoint(checkpoint, device)
    bundle = build_dataloaders(cfg, download=not no_download)
    loader = bundle.val if split == "val" else bundle.test

    logits, labels = collect_predictions(model, loader, device)
    report = build_report(logits, labels, split=split)
    click.echo(report.summary_line())

    if report_dir is None:
        return

    out = Path(report_dir)
    report.save_json(out / f"metrics_{split}.json")
    report.save_per_class_csv(out / f"per_class_{split}.csv")
    report.save_confusion(out / f"confusion_{split}.npy")

    if not no_figures:
        write_figures(report, logits, labels, out, history_path=history_path)

    click.echo(f"report written to {out}")


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--image", "image_path", type=click.Path(exists=True), required=True)
@click.option("--top-k", type=int, default=5)
def predict(checkpoint, image_path, top_k):
    """Print the top-k predicted flower classes for a single image."""
    device = resolve_device("auto")
    model, cfg = load_model_from_checkpoint(checkpoint, device)
    result = predict_image(
        model, image_path, device, image_size=cfg.data.image_size, top_k=top_k
    )
    for name, prob in zip(result.names, result.probs):
        click.echo(f"{prob:6.2%}  {name}")


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--image", "image_path", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(), default="gradcam.png",
              help="Where to write the image/heatmap panel.")
@click.option("--class-index", type=int, default=None,
              help="Explain this class instead of the predicted one.")
def explain(checkpoint, image_path, output, class_index):
    """Write a Grad-CAM panel showing where the model looked for one image."""
    device = resolve_device("auto")
    model, cfg = load_model_from_checkpoint(checkpoint, device)
    tensor = load_image_tensor(image_path, image_size=cfg.data.image_size, device=device)

    cam, label, prob = gradcam_explain(
        model, tensor, backbone_name=cfg.model.backbone, class_idx=class_index
    )
    title = f"{class_name(label)} ({prob:.1%})"
    save_gradcam_panel([tensor.detach().cpu()], [cam], [title], output)
    click.echo(f"{title} -> {output}")


if __name__ == "__main__":
    cli()
