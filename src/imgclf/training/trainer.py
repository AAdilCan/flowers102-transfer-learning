"""Training loop with cosine LR schedule, early stopping and checkpointing.

The same :func:`fit` drives both regimes. In ``linear_probe`` mode the caller
passes loaders of *cached features* and sets ``on_features=True`` so each batch
is fed straight to the classifier head; in ``finetune`` mode it passes image
loaders and the whole model runs per batch. Everything else — the optimiser, the
cosine schedule, early stopping and best-checkpoint tracking — is shared.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..logging_utils import get_logger
from ..models import TransferModel
from .checkpoint import save_checkpoint

logger = get_logger("trainer")


@dataclass
class EpochStats:
    """Loss and top-1 accuracy for one pass over a loader."""

    loss: float
    accuracy: float


@dataclass
class TrainHistory:
    """Per-epoch train/val curves plus the best val accuracy seen."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = -1

    def record(self, train: EpochStats, val: EpochStats) -> None:
        self.train_loss.append(train.loss)
        self.train_acc.append(train.accuracy)
        self.val_loss.append(val.loss)
        self.val_acc.append(val.accuracy)

    def to_dict(self) -> dict[str, object]:
        """Curves as plain lists, ready for JSON or the plotting helpers."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist the curves next to the checkpoint.

        Without this the per-epoch history dies with the process and the
        training-curve plot can only be produced by retraining.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("wrote training history -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainHistory":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


def _forward(model: TransferModel, batch: torch.Tensor, on_features: bool) -> torch.Tensor:
    """Route a batch through the head only (cached features) or the full model."""
    return model.classify_features(batch) if on_features else model(batch)


def _run_epoch(
    model: TransferModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    on_features: bool,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
) -> EpochStats:
    """One pass over ``loader``. Trains when ``optimizer`` is given, else evals."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = _forward(model, inputs, on_features)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optimizer.step()

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_n

    return EpochStats(loss=total_loss / seen, accuracy=correct / seen)


def fit(
    model: TransferModel,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    on_features: bool = False,
    checkpoint_path: str | Path | None = None,
) -> TrainHistory:
    """Train ``model`` and return its :class:`TrainHistory`.

    The best-by-val-accuracy weights are written to ``checkpoint_path`` (if
    given) and reloaded into ``model`` before returning, so the caller always
    holds the best epoch rather than the last one.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.optim.epochs,
        eta_min=cfg.optim.lr * cfg.optim.min_lr_factor,
    )

    history = TrainHistory()
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, cfg.optim.epochs + 1):
        train_stats = _run_epoch(
            model, train_loader, criterion, device,
            on_features=on_features, optimizer=optimizer,
            grad_clip=cfg.optim.grad_clip_norm,
        )
        val_stats = _run_epoch(
            model, val_loader, criterion, device,
            on_features=on_features, optimizer=None, grad_clip=None,
        )
        scheduler.step()
        history.record(train_stats, val_stats)

        logger.info(
            "epoch %02d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.2e",
            epoch, cfg.optim.epochs, train_stats.loss, train_stats.accuracy,
            val_stats.loss, val_stats.accuracy, scheduler.get_last_lr()[0],
        )

        if val_stats.accuracy > history.best_val_acc:
            history.best_val_acc = val_stats.accuracy
            history.best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path, model, cfg,
                    epoch=epoch, metrics={"val_acc": val_stats.accuracy},
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.optim.early_stopping_patience:
                logger.info(
                    "early stopping at epoch %d (no val gain for %d epochs)",
                    epoch, cfg.optim.early_stopping_patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(
        "training done | best val acc %.4f at epoch %d",
        history.best_val_acc, history.best_epoch,
    )
    return history
