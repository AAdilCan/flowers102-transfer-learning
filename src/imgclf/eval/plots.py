"""Figures for the evaluation harness.

Every function takes already-computed numbers and a destination path, writes a
PNG and returns that path. Matplotlib runs on the non-interactive Agg backend so
the plots can be produced from a script or CI without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..logging_utils import get_logger  # noqa: E402
from .metrics import EvaluationReport, top_k_accuracies  # noqa: E402

logger = get_logger("plots")

_DPI = 150


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote figure -> %s", path)
    return path


def plot_training_curves(history: Mapping[str, Sequence[float]], path: str | Path) -> Path:
    """Train/val loss and accuracy against epoch, side by side.

    Accepts anything dict-like with ``train_loss``/``val_loss``/``train_acc``/
    ``val_acc`` lists, i.e. ``TrainHistory.to_dict()`` or a loaded history JSON.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train", marker="o", ms=3)
    ax_loss.plot(epochs, history["val_loss"], label="val", marker="o", ms=3)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train", marker="o", ms=3)
    ax_acc.plot(epochs, history["val_acc"], label="val", marker="o", ms=3)
    best_epoch = history.get("best_epoch", -1)
    if isinstance(best_epoch, int) and best_epoch > 0:
        ax_acc.axvline(
            best_epoch, color="grey", ls="--", lw=1,
            label=f"best (epoch {best_epoch})",
        )
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("top-1 accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    return _save(fig, path)


def plot_confusion_matrix(
    confusion: np.ndarray, path: str | Path, *, normalize: bool = True
) -> Path:
    """Heatmap of the confusion matrix.

    With 102 classes the per-cell labels would be unreadable, so this is a plain
    heatmap: the diagonal should dominate and off-diagonal hot spots point at
    genuinely confusable flower pairs. Rows are normalised by support by default
    so that rare and common classes are comparable.
    """
    matrix = confusion.astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Empty rows (a class with no test samples) stay at zero instead of NaN.
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0 if normalize else None)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Row-normalised confusion matrix" if normalize else "Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class" if normalize else "count")
    return _save(fig, path)


def plot_worst_classes(
    report: EvaluationReport, path: str | Path, *, n: int = 20
) -> Path:
    """Horizontal bars for the ``n`` lowest-recall classes.

    This is the plot that actually drives further work: it names the flowers the
    model cannot tell apart instead of hiding them in an average.
    """
    worst = report.worst_classes(n)[::-1]  # reverse so the worst sits on top
    names = [f"{pc.name} (n={pc.support})" for pc in worst]
    recalls = [pc.recall for pc in worst]

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(worst) + 1.6))
    ax.barh(names, recalls, color="#c0504d")
    ax.axvline(report.balanced_accuracy, color="grey", ls="--", lw=1,
               label=f"balanced acc {report.balanced_accuracy:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall")
    ax.set_title(f"{n} weakest classes ({report.split} split)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_topk_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
    *,
    max_k: int = 10,
) -> Path:
    """Accuracy as a function of k — how far down the ranking the truth sits."""
    ks = tuple(range(1, max_k + 1))
    scores = top_k_accuracies(logits, labels, ks=ks)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [scores[k] for k in ks], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("top-k accuracy")
    ax.set_title("Accuracy vs k")
    ax.set_xticks(list(ks))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_reliability(
    confidence: np.ndarray, correct: np.ndarray, path: str | Path, *, n_bins: int = 15
) -> Path:
    """Reliability diagram: accuracy per confidence bin against the diagonal.

    Bars below the diagonal mean the model is overconfident, above means it is
    underconfident. Pairs with the ECE number in the metrics report.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.digitize(confidence, edges, right=True) - 1, 0, n_bins - 1)

    accs = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            accs[b] = correct[mask].mean()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.bar(centres, np.nan_to_num(accs), width=1.0 / n_bins * 0.9,
           color="#4f81bd", label="observed accuracy")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return _save(fig, path)
