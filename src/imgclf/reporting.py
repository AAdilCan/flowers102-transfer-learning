"""Turn an :class:`~imgclf.eval.metrics.EvaluationReport` into files on disk.

Kept separate from :mod:`imgclf.eval` so the metric code stays free of any
filesystem layout decisions: this module owns the "what gets written where"
convention that the CLI and the experiment script both follow.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .eval import (
    EvaluationReport,
    plot_confusion_matrix,
    plot_reliability,
    plot_topk_curve,
    plot_training_curves,
    plot_worst_classes,
)
from .logging_utils import get_logger
from .training import TrainHistory

logger = get_logger("reporting")


def write_figures(
    report: EvaluationReport,
    logits: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str | Path,
    *,
    history_path: str | Path | None = None,
    prefix: str = "",
) -> list[Path]:
    """Write the standard figure set for one evaluated split.

    ``prefix`` namespaces the filenames so several runs can share one directory
    (e.g. ``resnet18_probe_confusion_test.png``).
    """
    out_dir = Path(out_dir)
    split = report.split
    stem = f"{prefix}_" if prefix else ""
    paths = [
        plot_confusion_matrix(report.confusion, out_dir / f"{stem}confusion_{split}.png"),
        plot_worst_classes(report, out_dir / f"{stem}worst_classes_{split}.png"),
        plot_topk_curve(logits, labels, out_dir / f"{stem}topk_{split}.png"),
    ]

    probs = logits.softmax(dim=1)
    confidence = probs.max(dim=1).values.cpu().numpy()
    correct = (logits.argmax(dim=1) == labels).cpu().numpy().astype(float)
    paths.append(
        plot_reliability(confidence, correct, out_dir / f"{stem}reliability_{split}.png")
    )

    if history_path is not None:
        history = TrainHistory.load(history_path)
        paths.append(
            plot_training_curves(history.to_dict(), out_dir / f"{stem}training_curves.png")
        )

    logger.info("wrote %d figures to %s", len(paths), out_dir)
    return paths
