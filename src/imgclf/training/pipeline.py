"""High-level training orchestration.

Ties the pieces together so the CLI stays thin: seed, device, data, model, then
either the cached linear-probe path or end-to-end fine-tuning, with the config
persisted alongside the checkpoint for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..data import build_cache_loader, build_dataloaders
from ..logging_utils import get_logger
from ..models import build_model
from ..utils import resolve_device, set_seed
from .features import build_feature_loader, extract_features
from .trainer import TrainHistory, fit

logger = get_logger("pipeline")


def train_from_config(cfg: Config, *, download: bool = True) -> TrainHistory:
    """Run the full training pipeline described by ``cfg``.

    In ``linear_probe`` mode the frozen backbone features are cached once and the
    head is trained on them; in ``finetune`` mode the whole network trains on the
    image loaders. The best checkpoint lands at ``<output_dir>/best.pt`` and the
    resolved config next to it.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("device=%s | backbone=%s | mode=%s", device, cfg.model.backbone, cfg.model.mode)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(output_dir / "config.yaml")
    checkpoint_path = output_dir / "best.pt"

    bundle = build_dataloaders(cfg, download=download)
    model = build_model(cfg)

    if cfg.model.mode == "linear_probe":
        # Cache from deterministic, non-shuffled, non-dropping loaders rather
        # than reusing bundle.train: see build_cache_loader for why.
        logger.info("caching backbone features for linear probe")
        train_cache = build_cache_loader(cfg, "train", download=download)
        val_cache = build_cache_loader(cfg, "val", download=download)
        train_feats, train_labels = extract_features(model, train_cache, device)
        val_feats, val_labels = extract_features(model, val_cache, device)
        train_loader = build_feature_loader(
            train_feats, train_labels, batch_size=cfg.data.batch_size, shuffle=True
        )
        val_loader = build_feature_loader(
            val_feats, val_labels, batch_size=cfg.data.batch_size, shuffle=False
        )
        history = fit(
            model, cfg, train_loader, val_loader, device,
            on_features=True, checkpoint_path=checkpoint_path,
        )
    else:
        history = fit(
            model, cfg, bundle.train, bundle.val, device,
            on_features=False, checkpoint_path=checkpoint_path,
        )

    history.save(output_dir / "history.json")
    return history
