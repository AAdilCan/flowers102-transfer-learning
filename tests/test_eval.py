"""Tests for the metrics and plotting harness.

Metrics are checked against hand-computable cases rather than against another
implementation, so a silent change in averaging or in how empty classes are
handled shows up here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from imgclf.eval import (
    bootstrap_accuracy_ci,
    build_report,
    comparison_table,
    expected_calibration_error,
    plot_confusion_matrix,
    plot_reliability,
    plot_topk_curve,
    plot_training_curves,
    plot_worst_classes,
    top_k_accuracies,
)


def _logits_for(preds: list[int], num_classes: int = 4, margin: float = 5.0) -> torch.Tensor:
    """One-hot-ish logits that argmax to ``preds``."""
    logits = torch.zeros(len(preds), num_classes)
    for row, p in enumerate(preds):
        logits[row, p] = margin
    return logits


def test_perfect_predictions_score_one() -> None:
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    report = build_report(_logits_for(labels.tolist()), labels, split="test", n_bootstrap=100)

    assert report.top1 == pytest.approx(1.0)
    assert report.top5 == pytest.approx(1.0)
    assert report.balanced_accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.n_samples == 6
    assert np.trace(report.confusion) == 6


def test_per_class_metrics_match_hand_computation() -> None:
    # Class 0: 2 samples, 1 predicted correctly -> recall 0.5.
    # Class 1: 2 samples, both correct, but one class-0 sample also predicted
    # as 1 -> precision 2/3, recall 1.0.
    labels = torch.tensor([0, 0, 1, 1])
    report = build_report(
        _logits_for([0, 1, 1, 1], num_classes=3), labels, split="val", n_bootstrap=100
    )
    by_label = {pc.label: pc for pc in report.per_class}

    assert by_label[0].recall == pytest.approx(0.5)
    assert by_label[0].precision == pytest.approx(1.0)
    assert by_label[1].recall == pytest.approx(1.0)
    assert by_label[1].precision == pytest.approx(2 / 3)
    # Class 2 never appears and is never predicted: it must not be averaged in.
    assert by_label[2].support == 0
    assert report.balanced_accuracy == pytest.approx(0.75)
    assert report.top1 == pytest.approx(0.75)


def test_absent_classes_excluded_from_macro_averages() -> None:
    labels = torch.tensor([0, 0])
    report = build_report(_logits_for([0, 0], num_classes=50), labels, split="test", n_bootstrap=50)
    # Averaging over all 50 classes would give 1/50; only class 0 is present.
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.balanced_accuracy == pytest.approx(1.0)


def test_top_k_accuracies_are_monotonic() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 20)
    labels = torch.randint(0, 20, (64,))
    scores = top_k_accuracies(logits, labels, ks=(1, 3, 5, 10))
    values = [scores[k] for k in (1, 3, 5, 10)]
    assert values == sorted(values)
    assert set(scores) == {1, 3, 5, 10}


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    correct = np.array([1.0] * 70 + [0.0] * 30)
    low, high = bootstrap_accuracy_ci(correct, n_boot=1000, seed=1)
    assert low < correct.mean() < high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_is_seed_stable() -> None:
    correct = np.array([1.0, 0.0] * 50)
    assert bootstrap_accuracy_ci(correct, n_boot=200, seed=3) == bootstrap_accuracy_ci(
        correct, n_boot=200, seed=3
    )


def test_ece_zero_when_confidence_matches_accuracy() -> None:
    # 100 predictions at exactly 0.9 confidence, 90 of them correct.
    confidence = np.full(100, 0.9)
    correct = np.array([1.0] * 90 + [0.0] * 10)
    assert expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_penalises_overconfidence() -> None:
    confidence = np.full(100, 0.99)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(0.49, abs=1e-6)


def test_build_report_validates_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        build_report(torch.zeros(4), torch.zeros(4, dtype=torch.long), split="test")
    with pytest.raises(ValueError, match="mismatch"):
        build_report(torch.zeros(4, 3), torch.zeros(5, dtype=torch.long), split="test")


def test_report_artifacts_round_trip(tmp_path) -> None:
    labels = torch.tensor([0, 1, 2, 0])
    report = build_report(
        _logits_for([0, 1, 2, 1], num_classes=3), labels, split="test", n_bootstrap=100
    )

    json_path = tmp_path / "metrics.json"
    report.save_json(json_path)
    payload = json.loads(json_path.read_text())
    assert payload["split"] == "test"
    assert payload["top1"] == pytest.approx(0.75)
    assert len(payload["per_class"]) == 3
    assert "confusion" not in payload

    csv_path = tmp_path / "per_class.csv"
    report.save_per_class_csv(csv_path)
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0].startswith("label,name,support")
    assert len(rows) == 4  # header + 3 classes

    npy_path = tmp_path / "confusion.npy"
    report.save_confusion(npy_path)
    assert np.load(npy_path).shape == (3, 3)


def test_worst_and_best_classes_are_ordered() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    report = build_report(
        _logits_for([0, 0, 1, 0, 2, 0], num_classes=3), labels, split="test", n_bootstrap=50
    )
    worst = report.worst_classes(3)
    assert [pc.recall for pc in worst] == sorted(pc.recall for pc in worst)
    assert worst[0].label in (1, 2)
    assert report.best_classes(1)[0].label == 0


def test_comparison_table_lists_every_run() -> None:
    labels = torch.tensor([0, 1])
    a = build_report(_logits_for([0, 1], num_classes=2), labels, split="test", n_bootstrap=50)
    b = build_report(_logits_for([1, 1], num_classes=2), labels, split="test", n_bootstrap=50)
    table = comparison_table({"run_a": a, "run_b": b})
    assert "run_a" in table and "run_b" in table
    assert table.count("\n") == 4  # header, separator, two rows


def test_plots_write_files(tmp_path) -> None:
    torch.manual_seed(0)
    labels = torch.randint(0, 5, (40,))
    logits = torch.randn(40, 5)
    report = build_report(logits, labels, split="test", n_bootstrap=50)

    assert plot_confusion_matrix(report.confusion, tmp_path / "cm.png").exists()
    assert plot_worst_classes(report, tmp_path / "worst.png", n=3).exists()
    assert plot_topk_curve(logits, labels, tmp_path / "topk.png", max_k=4).exists()

    confidence = logits.softmax(dim=1).max(dim=1).values.numpy()
    correct = (logits.argmax(dim=1) == labels).numpy().astype(float)
    assert plot_reliability(confidence, correct, tmp_path / "rel.png").exists()

    history = {
        "train_loss": [2.0, 1.0], "val_loss": [2.1, 1.2],
        "train_acc": [0.1, 0.5], "val_acc": [0.1, 0.4], "best_epoch": 2,
    }
    assert plot_training_curves(history, tmp_path / "curves.png").exists()


def test_confusion_plot_handles_empty_rows(tmp_path) -> None:
    """A class with no test samples gives a zero row; normalising must not NaN."""
    confusion = np.zeros((3, 3), dtype=int)
    confusion[0, 0] = 5
    path = plot_confusion_matrix(confusion, tmp_path / "cm.png", normalize=True)
    assert path.exists()
