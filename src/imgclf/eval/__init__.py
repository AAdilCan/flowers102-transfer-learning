"""Evaluation harness: metrics, reports and figures."""

from .metrics import (
    EvaluationReport,
    PerClassMetrics,
    bootstrap_accuracy_ci,
    build_report,
    comparison_table,
    expected_calibration_error,
    top_k_accuracies,
)
from .plots import (
    plot_confusion_matrix,
    plot_reliability,
    plot_topk_curve,
    plot_training_curves,
    plot_worst_classes,
)

__all__ = [
    "EvaluationReport",
    "PerClassMetrics",
    "build_report",
    "comparison_table",
    "top_k_accuracies",
    "bootstrap_accuracy_ci",
    "expected_calibration_error",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_worst_classes",
    "plot_topk_curve",
    "plot_reliability",
]
