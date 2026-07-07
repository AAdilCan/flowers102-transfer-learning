# Flowers-102 Transfer Learning

A transfer-learning image classifier for the Oxford Flowers-102 dataset, built
in PyTorch. The point of the project is to compare two ways of reusing an
ImageNet-pretrained backbone on a dataset with very little labelled data
(only 10 training images per class):

- **Linear probing** — freeze the backbone, cache its features once, and train
  a lightweight classifier head. Fast enough to run comfortably on a CPU.
- **Fine-tuning** — unfreeze the whole network and train end-to-end with
  augmentation and a cosine learning-rate schedule.

The repo covers the full loop: data loading and augmentation, backbone
selection (ResNet-18/50, EfficientNet-B0), training with early stopping and
checkpointing, top-k / per-class evaluation, and Grad-CAM visualisations to
sanity-check what the model attends to.

## Why this dataset

Flowers-102 ships an unusual split: 1,020 train and 1,020 validation images
(10 per class) against 6,149 test images. That scarcity is deliberate here —
it is the regime where transfer learning earns its keep, and it makes the
linear-probe-vs-fine-tune comparison interesting rather than a formality.

## Status

Work in progress — built incrementally. Current scaffold:

```
src/imgclf/
├── config.py          # typed, YAML-serialisable run configuration
├── logging_utils.py   # shared logger
├── data/              # dataset, transforms, loaders  (in progress)
├── models/            # backbone factory + heads       (planned)
├── training/          # training loop, feature cache   (planned)
├── eval/              # metrics + plots                (planned)
└── interpret/         # Grad-CAM                        (planned)
scripts/explore_data.py
```

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/explore_data.py --root data
```

The dataset (~330 MB) downloads automatically to `data/` on first run.

## License

MIT
