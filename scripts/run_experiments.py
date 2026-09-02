"""Train and evaluate every backbone/mode combination, then write the results.

This is the script behind the results table in the README: it runs each
experiment end to end, scores the best checkpoint on the held-out test split,
writes per-run metrics and figures under ``reports/<name>/`` and finally a
single ``reports/results.md`` comparing them.

    PYTHONPATH=src python scripts/run_experiments.py --data-root data
    PYTHONPATH=src python scripts/run_experiments.py --only resnet18_probe

Runs are resumable: an experiment whose checkpoint already exists is evaluated
rather than retrained unless ``--force`` is passed, so a long CPU sweep can be
interrupted and picked back up.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from imgclf.config import Config
from imgclf.data import build_dataloaders
from imgclf.eval import build_report, comparison_table
from imgclf.inference import collect_predictions, load_model_from_checkpoint
from imgclf.logging_utils import configure_logging, get_logger
from imgclf.models import build_model
from imgclf.reporting import write_figures
from imgclf.training import TrainHistory, train_from_config
from imgclf.utils import count_parameters, resolve_device

logger = get_logger("experiments")

# Each entry is a partial config merged onto the defaults. Fine-tuning uses a
# much smaller LR than the probe: 1e-3 into a pretrained backbone erases the
# ImageNet features in the first few steps, which on 10 images per class is
# unrecoverable. It also gets fewer epochs simply because every epoch costs a
# full backward pass through the backbone on CPU.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "resnet18_probe": {
        "model": {"backbone": "resnet18", "mode": "linear_probe"},
        "optim": {"epochs": 40, "lr": 1e-3, "early_stopping_patience": 10},
    },
    "resnet50_probe": {
        "model": {"backbone": "resnet50", "mode": "linear_probe"},
        "optim": {"epochs": 40, "lr": 1e-3, "early_stopping_patience": 10},
    },
    "efficientnet_b0_probe": {
        "model": {"backbone": "efficientnet_b0", "mode": "linear_probe"},
        "optim": {"epochs": 40, "lr": 1e-3, "early_stopping_patience": 10},
    },
    "resnet18_finetune": {
        "model": {"backbone": "resnet18", "mode": "finetune"},
        "optim": {"epochs": 12, "lr": 1e-4, "early_stopping_patience": 4},
    },
}


def build_config(name: str, args: argparse.Namespace) -> Config:
    """Merge one experiment's overrides onto the defaults."""
    raw: dict[str, Any] = json.loads(json.dumps(EXPERIMENTS[name]))  # deep copy
    raw.setdefault("data", {})["root"] = args.data_root
    raw["data"]["batch_size"] = args.batch_size
    raw["data"]["num_workers"] = args.num_workers
    raw["output_dir"] = str(Path(args.checkpoint_root) / name)
    raw["seed"] = args.seed
    if args.epochs is not None:
        raw.setdefault("optim", {})["epochs"] = args.epochs
    return Config.from_dict(raw)


def run_experiment(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Train (or reuse) one checkpoint, evaluate it and write its artefacts."""
    cfg = build_config(name, args)
    checkpoint = Path(cfg.output_dir) / "best.pt"
    history_path = Path(cfg.output_dir) / "history.json"

    train_seconds = 0.0
    if checkpoint.exists() and not args.force:
        logger.info("[%s] reusing existing checkpoint %s", name, checkpoint)
    else:
        logger.info("[%s] training %s / %s", name, cfg.model.backbone, cfg.model.mode)
        start = time.perf_counter()
        train_from_config(cfg, download=not args.no_download)
        train_seconds = time.perf_counter() - start
        logger.info("[%s] trained in %.1fs", name, train_seconds)

    device = resolve_device(args.device)
    model, ckpt_cfg = load_model_from_checkpoint(checkpoint, device)
    bundle = build_dataloaders(ckpt_cfg, download=not args.no_download)

    logits, labels = collect_predictions(model, bundle.test, device)
    report = build_report(logits, labels, split="test")

    out_dir = Path(args.report_root) / name
    report.save_json(out_dir / "metrics_test.json")
    report.save_per_class_csv(out_dir / "per_class_test.csv")
    report.save_confusion(out_dir / "confusion_test.npy")
    write_figures(
        report, logits, labels, out_dir,
        history_path=history_path if history_path.exists() else None,
    )

    history = TrainHistory.load(history_path) if history_path.exists() else None
    # Parameter count comes from a throwaway randomly-initialised copy so this
    # never triggers a weights download just to print a number.
    trainable = count_parameters(build_model(cfg, pretrained=False), trainable_only=True)

    return {
        "name": name,
        "backbone": cfg.model.backbone,
        "mode": cfg.model.mode,
        "trainable_params": trainable,
        "epochs_run": len(history.train_loss) if history else None,
        "best_epoch": history.best_epoch if history else None,
        "best_val_acc": history.best_val_acc if history else None,
        "train_seconds": round(train_seconds, 1) if train_seconds else None,
        "report": report,
    }


def write_results(results: list[dict[str, Any]], report_root: Path) -> None:
    """Write the markdown comparison table and a machine-readable summary."""
    reports = {r["name"]: r["report"] for r in results}

    lines = [
        "# Results",
        "",
        "Oxford Flowers-102, held-out **test** split (6,149 images), trained on the",
        "official 1,020-image train split with 1,020 images used for validation and",
        "model selection. Confidence intervals are 2,000-sample percentile bootstraps",
        "over the test set.",
        "",
        comparison_table(reports),
        "### Cost and training detail",
        "",
        "| run | trainable params | epochs run | best epoch | best val acc | train time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        val = f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "-"
        secs = f"{r['train_seconds']:.0f}" if r["train_seconds"] else "reused"
        lines.append(
            f"| {r['name']} | {r['trainable_params']:,} | {r['epochs_run'] or '-'} | "
            f"{r['best_epoch'] or '-'} | {val} | {secs} |"
        )

    lines += ["", "### Weakest classes (best run)", ""]
    best = max(results, key=lambda r: r["report"].top1)
    lines.append(f"Lowest-recall classes for `{best['name']}`:")
    lines.append("")
    lines.append("| class | support | recall | precision | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for pc in best["report"].worst_classes(10):
        lines.append(
            f"| {pc.name} | {pc.support} | {pc.recall:.3f} | "
            f"{pc.precision:.3f} | {pc.f1:.3f} |"
        )
    lines.append("")
    lines.append(f"Figures for each run are in `reports/<run>/`.")
    lines.append("")

    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "results.md").write_text("\n".join(lines), encoding="utf-8")

    summary = [
        {k: v for k, v in r.items() if k != "report"} | r["report"].to_dict()
        for r in results
    ]
    with (report_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("wrote %s", report_root / "results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--only", action="append", choices=sorted(EXPERIMENTS),
                        help="Run a subset; repeatable.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the per-experiment epoch count.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if a checkpoint already exists.")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    names = args.only or list(EXPERIMENTS)

    results = []
    for name in names:
        results.append(run_experiment(name, args))
        # Write after every run so an interrupted sweep still leaves a table.
        write_results(results, Path(args.report_root))

    print()
    for r in results:
        print(f"{r['name']:>22}  {r['report'].summary_line()}")


if __name__ == "__main__":
    main()
