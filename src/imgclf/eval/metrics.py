"""Classification metrics for the evaluation harness.

Everything here works on already-collected ``(logits, labels)`` tensors — see
:func:`imgclf.inference.collect_predictions` — so metrics can be recomputed
offline without touching the model or the dataset again.

Top-1 accuracy alone is a poor summary on Flowers-102: the test split is
class-imbalanced (20 to 238 images per class), so a model can look good while
failing the rare classes entirely. The report therefore also carries balanced
accuracy, per-class precision/recall/F1, a bootstrap confidence interval and a
calibration error.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from ..data.class_names import CLASS_NAMES
from ..inference import top_k_accuracy
from ..logging_utils import get_logger

logger = get_logger("eval")


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision/recall/F1 for a single class, plus how many samples it had."""

    label: int
    name: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationReport:
    """Everything measured on one split for one checkpoint.

    ``confusion`` is kept as a NumPy array rather than being folded into
    :meth:`to_dict`, because a 102x102 matrix belongs in a ``.npy``/plot, not in
    a summary JSON.
    """

    split: str
    n_samples: int
    num_classes: int
    top1: float
    top5: float
    top1_ci_low: float
    top1_ci_high: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_confidence: float
    expected_calibration_error: float
    per_class: list[PerClassMetrics]
    confusion: np.ndarray

    def to_dict(self) -> dict:
        """Summary metrics as plain JSON-serialisable types (no matrix)."""
        payload = {
            k: v
            for k, v in asdict(self).items()
            if k not in ("per_class", "confusion")
        }
        payload["per_class"] = [asdict(pc) for pc in self.per_class]
        return payload

    def worst_classes(self, n: int = 10) -> list[PerClassMetrics]:
        """The ``n`` classes with the lowest recall (ties broken by support)."""
        return sorted(self.per_class, key=lambda pc: (pc.recall, -pc.support))[:n]

    def best_classes(self, n: int = 10) -> list[PerClassMetrics]:
        return sorted(self.per_class, key=lambda pc: (-pc.f1, -pc.support))[:n]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("wrote metrics -> %s", path)

    def save_per_class_csv(self, path: str | Path) -> None:
        """Dump the full 102-row per-class table; too long for the markdown."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["label", "name", "support", "precision", "recall", "f1"]
            )
            writer.writeheader()
            for pc in sorted(self.per_class, key=lambda p: p.label):
                writer.writerow(asdict(pc))
        logger.info("wrote per-class table -> %s", path)

    def save_confusion(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.confusion)

    def summary_line(self) -> str:
        return (
            f"{self.split}: top-1 {self.top1:.4f} "
            f"[{self.top1_ci_low:.4f}, {self.top1_ci_high:.4f}] | "
            f"top-5 {self.top5:.4f} | balanced {self.balanced_accuracy:.4f} | "
            f"macro-F1 {self.macro_f1:.4f} | ECE {self.expected_calibration_error:.4f} "
            f"| n={self.n_samples}"
        )


def top_k_accuracies(
    logits: torch.Tensor, labels: torch.Tensor, ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[int, float]:
    """Top-k accuracy for several ``k`` at once (for the accuracy-vs-k curve)."""
    return {k: top_k_accuracy(logits, labels, k=k) for k in ks}


def bootstrap_accuracy_ci(
    correct: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 correctness vector.

    A single accuracy number hides how much of the gap between two models is
    sampling noise. Resampling the test set gives an honest interval; with
    ~6k test images the interval is roughly +/- 1 point, which is enough to say
    whether two backbones are actually different.
    """
    if correct.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, correct.size, size=(n_boot, correct.size))
    means = correct[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def expected_calibration_error(
    probs: np.ndarray, correct: np.ndarray, *, n_bins: int = 15
) -> float:
    """Equal-width-bin ECE over the predicted-class confidence.

    Bins predictions by confidence and averages ``|accuracy - confidence|``
    weighted by bin size. A model that says "90% sure" should be right 90% of
    the time; label smoothing and a tiny training set both push this around, so
    it is worth reporting next to accuracy.
    """
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]; confidences of exactly 1.0 fall in the last bin.
    bins = np.clip(np.digitize(probs, edges, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        weight = mask.mean()
        ece += weight * abs(correct[mask].mean() - probs[mask].mean())
    return float(ece)


def build_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    split: str,
    class_names: tuple[str, ...] = CLASS_NAMES,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> EvaluationReport:
    """Compute every metric for one split from stacked logits and labels."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (n, num_classes), got {tuple(logits.shape)}")
    if logits.size(0) != labels.size(0):
        raise ValueError(
            f"logits/labels length mismatch: {logits.size(0)} vs {labels.size(0)}"
        )

    num_classes = logits.size(1)
    probs = logits.softmax(dim=1)
    preds = logits.argmax(dim=1)

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    correct = (y_pred == y_true).astype(np.float64)
    confidence = probs.max(dim=1).values.cpu().numpy()

    label_range = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_range, zero_division=0
    )
    per_class = [
        PerClassMetrics(
            label=i,
            name=class_names[i] if i < len(class_names) else f"class_{i}",
            support=int(support[i]),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
        )
        for i in label_range
    ]

    # Balanced accuracy = mean recall over the classes that actually appear,
    # so absent classes do not drag the average toward zero.
    present = support > 0
    balanced = float(recall[present].mean()) if present.any() else float("nan")

    ci_low, ci_high = bootstrap_accuracy_ci(correct, n_boot=n_bootstrap, seed=seed)

    report = EvaluationReport(
        split=split,
        n_samples=int(labels.numel()),
        num_classes=num_classes,
        top1=top_k_accuracy(logits, labels, k=1),
        top5=top_k_accuracy(logits, labels, k=min(5, num_classes)),
        top1_ci_low=ci_low,
        top1_ci_high=ci_high,
        balanced_accuracy=balanced,
        macro_precision=float(precision[present].mean()) if present.any() else float("nan"),
        macro_recall=balanced,
        macro_f1=float(f1[present].mean()) if present.any() else float("nan"),
        mean_confidence=float(confidence.mean()),
        expected_calibration_error=expected_calibration_error(confidence, correct),
        per_class=per_class,
        confusion=sk_confusion_matrix(y_true, y_pred, labels=label_range),
    )
    logger.info(report.summary_line())
    return report


def comparison_table(reports: dict[str, EvaluationReport]) -> str:
    """Markdown results table comparing several runs on the same split."""
    header = (
        "| run | top-1 | 95% CI | top-5 | balanced acc | macro-F1 | ECE |\n"
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = "".join(
        f"| {name} | {r.top1:.4f} | "
        f"[{r.top1_ci_low:.4f}, {r.top1_ci_high:.4f}] | {r.top5:.4f} | "
        f"{r.balanced_accuracy:.4f} | {r.macro_f1:.4f} | "
        f"{r.expected_calibration_error:.4f} |\n"
        for name, r in reports.items()
    )
    return header + rows
