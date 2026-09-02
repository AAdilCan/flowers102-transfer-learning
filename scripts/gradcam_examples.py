"""Build the Grad-CAM figure used in the README.

Scans part of the test split with a trained checkpoint, then explains the most
confident correct predictions alongside the most confident *wrong* ones. The
confident mistakes are the interesting half: they show whether a wrong answer
came from looking at the wrong thing (background, a neighbouring bloom) or from
two genuinely similar flowers.

    PYTHONPATH=src python scripts/gradcam_examples.py \\
        --checkpoint checkpoints/resnet18_probe/best.pt \\
        --output reports/gradcam_examples.png
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from imgclf.data import build_dataset
from imgclf.data.class_names import class_name
from imgclf.interpret import GradCAM, resolve_target_layer, save_gradcam_panel
from imgclf.inference import load_model_from_checkpoint
from imgclf.logging_utils import configure_logging, get_logger
from imgclf.utils import resolve_device

logger = get_logger("gradcam_examples")


@torch.no_grad()
def scan_predictions(
    model, dataset, device: torch.device, *, limit: int, batch_size: int
) -> list[tuple[int, int, int, float]]:
    """Return ``(index, true_label, predicted_label, confidence)`` per image."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[tuple[int, int, int, float]] = []
    index = 0
    for images, labels in loader:
        probs = model(images.to(device)).softmax(dim=1).cpu()
        confidence, preds = probs.max(dim=1)
        for true, pred, conf in zip(labels.tolist(), preds.tolist(), confidence.tolist()):
            rows.append((index, true, pred, conf))
            index += 1
        if index >= limit:
            break
    logger.info("scanned %d test images", len(rows))
    return rows


def pick_examples(
    rows: list[tuple[int, int, int, float]], n_correct: int, n_wrong: int
) -> list[tuple[int, int, int, float]]:
    """Most confident correct predictions first, then most confident mistakes."""
    correct = sorted((r for r in rows if r[1] == r[2]), key=lambda r: -r[3])[:n_correct]
    wrong = sorted((r for r in rows if r[1] != r[2]), key=lambda r: -r[3])[:n_wrong]
    if not wrong:
        logger.warning("no misclassified image in the scanned subset")
    return correct + wrong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="reports/gradcam_examples.png")
    parser.add_argument("--data-root", default=None,
                        help="Override the checkpoint's data root.")
    parser.add_argument("--n-correct", type=int, default=3)
    parser.add_argument("--n-wrong", type=int, default=3)
    parser.add_argument("--limit", type=int, default=1500,
                        help="How many test images to scan before choosing.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    device = resolve_device(args.device)
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    if args.data_root:
        cfg.data.root = args.data_root

    dataset = build_dataset(cfg, "test", download=False, augment=False)
    rows = scan_predictions(
        model, dataset, device, limit=args.limit, batch_size=args.batch_size
    )
    chosen = pick_examples(rows, args.n_correct, args.n_wrong)

    images, cams, titles = [], [], []
    target_layer = resolve_target_layer(model, cfg.model.backbone)
    with GradCAM(model, target_layer) as gradcam:
        for index, true, pred, conf in chosen:
            image = dataset[index][0].unsqueeze(0).to(device)
            # Explain the class the model actually chose, right or wrong.
            cam, _, _ = gradcam(image, class_idx=pred)
            images.append(image.detach().cpu())
            cams.append(cam)
            mark = "correct" if true == pred else "WRONG"
            titles.append(
                f"{mark}: true {class_name(true)} | pred {class_name(pred)} ({conf:.0%})"
            )

    save_gradcam_panel(images, cams, titles, args.output)
    print(f"wrote {args.output} ({len(images)} examples)")


if __name__ == "__main__":
    main()
