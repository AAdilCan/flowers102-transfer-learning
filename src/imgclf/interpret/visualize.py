"""Rendering helpers for Grad-CAM output."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..config import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402

logger = get_logger("interpret")


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Undo the ImageNet normalisation and return an ``(H, W, 3)`` array in [0, 1].

    Working back from the model input rather than re-reading the file keeps the
    heatmap pixel-aligned with exactly what the model saw (same resize and crop).
    """
    if image.dim() == 4:
        if image.size(0) != 1:
            raise ValueError("denormalize expects a single image")
        image = image[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = image.detach().cpu() * std + mean
    return restored.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_cam(image: torch.Tensor, cam: torch.Tensor, *, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over the de-normalised image; returns ``(H, W, 3)`` in [0, 1]."""
    base = denormalize(image)
    heat = plt.get_cmap("jet")(cam.detach().cpu().numpy())[..., :3]
    # Clip: float32 rounding can push the blend a hair past 1.0, which
    # matplotlib's imsave rejects outright.
    return np.clip((1 - alpha) * base + alpha * heat, 0.0, 1.0)


def save_gradcam_panel(
    images: Sequence[torch.Tensor],
    cams: Sequence[torch.Tensor],
    titles: Sequence[str],
    path: str | Path,
    *,
    alpha: float = 0.45,
) -> Path:
    """Grid of image / overlay pairs, one row per example.

    Left column is what the model saw, right column the same image under its
    Grad-CAM heatmap, so a reader can judge the attention without flipping
    between figures.
    """
    if not (len(images) == len(cams) == len(titles)):
        raise ValueError("images, cams and titles must be the same length")
    if not images:
        raise ValueError("nothing to plot")

    rows = len(images)
    fig, axes = plt.subplots(rows, 2, figsize=(6.2, 3.1 * rows), squeeze=False)
    for row, (image, cam, title) in enumerate(zip(images, cams, titles)):
        axes[row][0].imshow(denormalize(image))
        axes[row][0].set_title(title, fontsize=9, loc="left")
        axes[row][1].imshow(overlay_cam(image, cam, alpha=alpha))
        axes[row][1].set_title("Grad-CAM", fontsize=9, loc="left")
        for ax in axes[row]:
            ax.axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote Grad-CAM panel -> %s", path)
    return path
