"""Download Oxford Flowers-102 and print a short exploratory summary.

Run once during setup to materialise the dataset under ``data/`` and to sanity
check the class balance and image sizes before wiring up the training pipeline::

    PYTHONPATH=src python scripts/explore_data.py --root data
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
from torchvision.datasets import Flowers102

# The official split is unusual: only 10 images per class for train AND val,
# with the large remainder held out for test. That scarcity is exactly why
# transfer learning matters here, so it is worth surfacing early.
SPLITS = ("train", "val", "test")


def summarise_split(root: str, split: str) -> dict[str, object]:
    ds = Flowers102(root=root, split=split, download=True)
    # torchvision stores the integer label list on the private ``_labels``
    # attribute; reading it avoids decoding every image just to count classes.
    counts = Counter(int(label) for label in ds._labels)
    per_class = list(counts.values())
    return {
        "split": split,
        "n_images": len(ds),
        "n_classes": len(counts),
        "min_per_class": min(per_class),
        "max_per_class": max(per_class),
    }


def sample_image_sizes(root: str, split: str, n: int = 50) -> tuple[int, int, int, int]:
    """Return (min_w, min_h, max_w, max_h) across the first ``n`` images."""
    ds = Flowers102(root=root, split=split, download=True)
    widths, heights = [], []
    for i in range(min(n, len(ds))):
        img, _ = ds[i]
        if not isinstance(img, Image.Image):
            continue
        widths.append(img.width)
        heights.append(img.height)
    return min(widths), min(heights), max(widths), max(heights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Oxford Flowers-102.")
    parser.add_argument("--root", default="data", help="Dataset download root.")
    args = parser.parse_args()

    Path(args.root).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Oxford Flowers-102 summary")
    print("=" * 60)
    for split in SPLITS:
        info = summarise_split(args.root, split)
        print(
            f"{info['split']:>5}: {info['n_images']:>5} images | "
            f"{info['n_classes']:>3} classes | "
            f"per-class min={info['min_per_class']} max={info['max_per_class']}"
        )

    mn_w, mn_h, mx_w, mx_h = sample_image_sizes(args.root, "train")
    print("-" * 60)
    print(f"train image size range (first 50): "
          f"w[{mn_w}, {mx_w}]  h[{mn_h}, {mx_h}]")
    print("Images vary in size -> resize/crop transforms are required.")


if __name__ == "__main__":
    main()
