# Documentation

Technical notes on the Flowers-102 transfer-learning project: how it is put
together, what I measured, and which decisions I would defend in a review.

## 1. Overview

The task is 102-way flower classification from the Oxford Flowers-102 dataset.
What makes it interesting is the split: **10 training images per class**. That
is far too little to train a convolutional network from scratch, so the whole
project is about reusing an ImageNet-pretrained backbone well, and about
measuring the result honestly on the 6,149-image test split.

Two regimes are implemented and compared:

- **Linear probe** — freeze the backbone, run every image through it exactly
  once, cache the pooled feature vectors, and train a dropout+linear head on
  those cached tensors. Cheap enough to train in about two minutes on a CPU.
- **Fine-tuning** — unfreeze the whole network and train end to end at a much
  lower learning rate, with augmentation and a cosine schedule.

Everything runs on CPU. No GPU was used at any point, which constrained some of
the choices below and is called out where it did.

## 2. Architecture

```
src/imgclf/
├── config.py            typed, YAML-serialisable run configuration
├── logging_utils.py     one logger namespace for the whole package
├── utils.py             seeding, device resolution, parameter counting
├── inference.py         checkpoint -> model, batch scoring, single-image predict
├── reporting.py         which artefacts get written where
├── cli.py               train | evaluate | predict | explain
├── data/
│   ├── transforms.py    train (stochastic) and eval (deterministic) pipelines
│   ├── dataset.py       Flowers102 wrapper, loaders, feature-cache loader
│   └── class_names.py   102 readable class names indexed by label
├── models/
│   ├── backbones.py     resnet18 / resnet50 / efficientnet_b0, classifier stripped
│   └── classifier.py    ClassifierHead + TransferModel (probe / finetune modes)
├── training/
│   ├── features.py      frozen-backbone feature cache
│   ├── trainer.py       shared loop: cosine LR, early stopping, best-checkpoint
│   ├── checkpoint.py    self-describing checkpoint payloads
│   └── pipeline.py      orchestration behind `imgclf train`
├── eval/
│   ├── metrics.py       report: top-k, per-class, bootstrap CI, ECE
│   └── plots.py         confusion, weakest classes, top-k curve, reliability
└── interpret/
    ├── gradcam.py       hooked activations/gradients -> heatmap
    └── visualize.py     denormalise, overlay, panel figure
```

Data flow for the two regimes:

```
                    ┌──────────────┐
   images ─────────▶│   backbone   │─── pooled features ──┐
                    │  (frozen or  │                      ▼
                    │   trainable) │              ┌───────────────┐
                    └──────────────┘              │ ClassifierHead│──▶ logits
                            ▲                     └───────────────┘
                            │                             ▲
  linear probe: run ONCE, cache features ─────────────────┘
  finetune:     run EVERY batch, gradients flow back through the backbone
```

The single hinge that makes this work is in `src/imgclf/models/backbones.py`:
each backbone's final ImageNet classifier is replaced with `nn.Identity`, so the
backbone *is* the feature extractor. `TransferModel.extract_features` and
`TransferModel.classify_features` then expose the two halves separately, which
is what lets `training/features.py` cache the expensive half.

## 3. Data

- **Source**: Oxford Flowers-102 (Nilsback & Zisserman, 2008), pulled through
  `torchvision.datasets.Flowers102`; ~345 MB, auto-downloaded to `data/`.
- **Splits** (the official ones, unchanged): 1,020 train / 1,020 val / 6,149
  test. Train and val hold exactly 10 images per class; the test split is
  imbalanced, from 20 images (`canterbury bells`) to 238 (`petunia`).
- **Images**: variable size, roughly 500–850 px per side, so a resize is
  mandatory rather than optional.

Preprocessing (`src/imgclf/data/transforms.py`):

| stage | train | val / test |
| --- | --- | --- |
| geometry | `RandomResizedCrop(224, scale=(0.6, 1.0))` | `Resize(256)` + `CenterCrop(224)` |
| flip | horizontal, p=0.5 | — |
| rotation | ±20° | — |
| colour | jitter 0.2 (brightness/contrast/saturation) | — |
| normalise | ImageNet mean/std | ImageNet mean/std |

The ImageNet statistics are not decoration: the pretrained weights were fitted
under that normalisation, and using anything else quietly degrades the features
before training even starts.

I kept the official validation split for model selection rather than merging it
into training. Merging would double the training data to 20 images per class —
tempting — but then early stopping and best-epoch selection would have no honest
signal, and every number in this document would be selected on the test set.

## 4. Methodology

**Backbones.** ResNet-18 (512-d features), ResNet-50 (2048-d) and
EfficientNet-B0 (1280-d), all ImageNet-pretrained. They cover a 4x range in
feature width and a much wider range in inference cost, which is the axis I
cared about given a CPU budget.

**Head.** Dropout(0.2) + a single `Linear(feature_dim, 102)`. With 1,020
training images, anything deeper mostly adds variance: a ResNet-18 probe head is
52k parameters against 11.2M frozen ones, and even that overfits to ~99% train
accuracy within 20 epochs.

**Optimisation.** AdamW, cosine annealing to 1% of the base LR, label smoothing
0.1, early stopping on validation accuracy. The probe trains at `lr=1e-3`;
fine-tuning uses `lr=1e-4`, because 1e-3 into a pretrained backbone destroys the
ImageNet features in the first few steps and there is not enough data to
re-learn them.

**Feature caching.** For a frozen backbone the features are a fixed function of
the image, so `training/features.py` computes them once and the head then trains
on cached tensors. This is what turns a 40-epoch probe into a ~2-minute CPU job:
one backbone pass over 2,040 images instead of forty.

Two details in `data/dataset.build_cache_loader` matter more than they look:

1. **Caching uses the deterministic eval transform, not augmentation.** Caching
   augmented features would freeze exactly one random crop per image for the
   entire run — strictly worse than no augmentation, because the single sample
   might be a bad crop and it can never be resampled.
2. **`drop_last=False`.** The training loader drops the final partial batch to
   keep BatchNorm statistics sane, but reusing it for caching would silently
   discard up to `batch_size - 1` of the 1,020 training images. At batch size 32
   that is 12 images, more than a whole class.

**Metrics.** `eval/metrics.py` reports top-1 and top-5, per-class
precision/recall/F1, balanced accuracy (mean per-class recall), a 2,000-sample
bootstrap CI on top-1, and expected calibration error. Top-1 alone is misleading
here because the test split is imbalanced: a model can score well while failing
the rare classes, and balanced accuracy is what exposes that.

**Interpretability.** Grad-CAM (`interpret/gradcam.py`) hooks the last spatial
block (`layer4` for the ResNets, `features` for EfficientNet), weights the
activation maps by the spatially averaged gradient of the class score, and
upsamples the positive part to the input resolution.

## 5. Results

Held-out **test** split (6,149 images). Every model was selected on the
validation split by top-1 accuracy; the test split was scored once per run.
Intervals are 2,000-sample percentile bootstraps over the test set.

| run | top-1 | 95% CI | top-5 | balanced acc | macro-F1 | ECE |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| resnet18_probe | 0.8310 | [0.8216, 0.8401] | 0.9550 | 0.8511 | 0.8269 | 0.3303 |
| resnet50_probe | 0.8528 | [0.8440, 0.8621] | 0.9632 | 0.8640 | 0.8509 | 0.3188 |
| **efficientnet_b0_probe** | **0.8697** | [0.8611, 0.8784] | 0.9645 | **0.8831** | **0.8631** | 0.3415 |
| resnet18_finetune | 0.8413 | [0.8314, 0.8504] | 0.9489 | 0.8687 | 0.8412 | 0.3627 |

| run | trainable params | epochs | best epoch | best val acc | train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| resnet18_probe | 52,326 | 40 | 32 | 0.8627 | 132 s |
| resnet50_probe | 208,998 | 36 | 26 | 0.8686 | 311 s |
| efficientnet_b0_probe | 130,662 | 39 | 29 | 0.8990 | 189 s |
| resnet18_finetune | 11,228,838 | 12 | 11 | 0.8735 | 1,510 s |

Figures per run live in `reports/<run>/`: training curves, row-normalised
confusion matrix, the twenty weakest classes, an accuracy-vs-k curve and a
reliability diagram. `reports/results.md` is regenerated by the sweep script.

### What the numbers say

**Picking a better backbone beat fine-tuning a worse one — for an eighth of the
compute.** The EfficientNet-B0 probe reaches 87.0% in 189 seconds while the
ResNet-18 fine-tune reaches 84.1% in 1,510 seconds, and their confidence
intervals are nowhere near each other. Given a fixed CPU budget, the first thing
to spend it on is the feature extractor, not the training regime.

**Fine-tuning ResNet-18 did not clearly beat probing it.** 0.8413 against 0.8310
looks like a win, but the intervals ([0.8314, 0.8504] and [0.8216, 0.8401])
overlap, so this run does not establish a real difference — for 215x the
trainable parameters and 11x the wall-clock time. It is worth noticing where the
gain does show up: balanced accuracy moves further than top-1 (0.8511 to 0.8687),
so what fine-tuning bought was mostly on the smaller classes.

The fine-tune result is a floor rather than a verdict, though:
`reports/resnet18_finetune/training_curves.png` shows validation accuracy still
climbing at epoch 12, where the budget ran out. The honest statement is that
fine-tuning did not pay for itself *within this compute budget*, not that it
cannot.

**Balanced accuracy is above top-1 in every single run.** That inverts the usual
imbalanced-data story: the model is doing *worse* on the biggest classes, not
better. `petunia` has 238 test images — more than any other class — and only
0.592 recall in the best run. Big classes here are big because the flower is
visually diverse, not because it is easy.

**Top-5 sits at 95–96% while top-1 is in the mid-80s.** Errors are near-misses
inside a small candidate set rather than wild guesses, which is what the
Grad-CAM panel shows too: the confident mistakes are on genuinely similar
flowers (`globe-flower` vs `buttercup`, `artichoke` vs `spear thistle`).

**Every model is badly calibrated, in the same direction.** ECE runs 0.32–0.36,
and the direction is *under*confidence: the EfficientNet probe averages 0.53
confidence while being right 87% of the time. The cause is the label smoothing
of 0.1 in the loss — it deliberately stops the network from putting mass near
1.0, and nothing afterwards undoes that. The reliability diagrams in
`reports/<run>/reliability_test.png` show the bars sitting above the diagonal
everywhere. Any downstream use of these probabilities as confidence would be
wrong, and fixing it is the first item in section 8.

### Weakest classes (EfficientNet-B0 probe)

| class | test images | recall | precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| sweet pea | 36 | 0.528 | 0.452 | 0.487 |
| sword lily | 110 | 0.545 | 0.909 | 0.682 |
| canterbury bells | 20 | 0.550 | 0.379 | 0.449 |
| petunia | 238 | 0.592 | 0.865 | 0.703 |
| camellia | 71 | 0.620 | 0.786 | 0.693 |

`sword lily` is the interesting row: 0.909 precision against 0.545 recall means
the model is reliable when it commits, but it lets nearly half of them go to
another class. That is a decision-threshold problem, not a feature problem, and
it is invisible in any top-1 average. The full table for each run is in
`reports/<run>/per_class_test.csv`.

## 6. Tradeoffs and decisions

**Linear probe as the default, not the fallback.** The probe is the default
`mode` in `config.py`. On 10 images per class it is not obviously the weaker
option — the frozen features are already good, and there is far less to overfit.
It also happens to be the only regime that trains comfortably on CPU, so the
default is the configuration a reader can actually reproduce.

**Caching features instead of a `torch.no_grad()` fast path.** I could have kept
one code path and simply skipped gradients for the frozen backbone. Caching is
more code (`features.py`, `build_cache_loader`, the `on_features` flag through
`trainer.fit`) but it removes ~39 redundant passes over the dataset. The cost is
that the probe cannot use augmentation at all — see the honest limitation below.

**One `fit` function for both regimes.** `trainer.fit` takes an `on_features`
flag and routes each batch either to the head alone or through the whole model.
Two separate training loops would have drifted apart on early stopping and
checkpoint semantics; one loop with one branch keeps them identical by
construction.

**Self-describing checkpoints.** `save_checkpoint` stores the config next to the
weights, so `evaluate`, `predict` and `explain` rebuild the exact architecture
from the file and never need `--backbone` on the command line. It also means a
checkpoint remains loadable after the defaults in `config.py` change.

**Overriding `TransferModel.train()`.** `nn.Module.train()` would flip a frozen
backbone back into training mode, silently letting its BatchNorm layers update
their running statistics from flower images. Freezing `requires_grad` alone does
not prevent that; the override in `models/classifier.py` does.

### Honest limitations

- **The probe gets no augmentation.** Caching and augmentation are mutually
  exclusive by construction, and on 10 images per class augmentation is exactly
  the sort of thing that should help. The fine-tuning runs do use it, which
  makes that comparison a comparison of two bundles (regime + augmentation)
  rather than a clean ablation of the regime alone.
- **CPU-bound experiment design.** Fine-tuning got 12 epochs against the probe's
  40, and no hyperparameter search was run for either. The training curves show
  the fine-tune had not converged when it stopped — validation accuracy was
  still rising at the final epoch — so its number is a floor, not the ceiling of
  the method. A GPU and 60 epochs would very likely reorder the table.
- **Single seed.** Every run uses `seed=42`. The bootstrap CIs quantify test-set
  sampling noise, not seed-to-seed training variance, and those are different
  things.
- **The test split was used once per run, at the end.** Model selection is on
  validation only — but I did look at test numbers while writing this document,
  so treat the *relative ranking* as sound and the absolute numbers as very
  mildly optimistic.

## 7. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The package resolves from `src/` via `pythonpath` in `pyproject.toml`, so
`PYTHONPATH=src` is all that scripts need (an editable install works too).

Explore the dataset (downloads ~345 MB on first run):

```bash
PYTHONPATH=src python scripts/explore_data.py --root data
```

Train a linear probe:

```bash
PYTHONPATH=src python -m imgclf.cli train \
    --backbone resnet18 --mode linear_probe --epochs 40 \
    --output-dir checkpoints/resnet18_probe
```

Fine-tune end to end instead:

```bash
PYTHONPATH=src python -m imgclf.cli train \
    --backbone resnet18 --mode finetune --epochs 12 --lr 1e-4 \
    --output-dir checkpoints/resnet18_finetune
```

Evaluate a checkpoint and write the full report (metrics JSON, per-class CSV,
confusion matrix, figures):

```bash
PYTHONPATH=src python -m imgclf.cli evaluate \
    --checkpoint checkpoints/resnet18_probe/best.pt --split test \
    --report-dir reports/resnet18_probe \
    --history checkpoints/resnet18_probe/history.json
```

Classify one image, and see where the model looked:

```bash
PYTHONPATH=src python -m imgclf.cli predict \
    --checkpoint checkpoints/resnet18_probe/best.pt --image path/to/flower.jpg

PYTHONPATH=src python -m imgclf.cli explain \
    --checkpoint checkpoints/resnet18_probe/best.pt \
    --image path/to/flower.jpg --output gradcam.png
```

Reproduce every run in the results table, then rebuild the Grad-CAM figure:

```bash
PYTHONPATH=src python scripts/run_experiments.py --data-root data
PYTHONPATH=src python scripts/gradcam_examples.py \
    --checkpoint checkpoints/resnet18_probe/best.pt \
    --output reports/gradcam_examples.png
```

Tests (no dataset download required — `tests/conftest.py` fakes Flowers-102 at
the torchvision boundary):

```bash
PYTHONPATH=src python -m pytest -q
```

## 8. How to extend

- **Fix the calibration.** Fit a single temperature on the validation logits and
  divide the test logits by it. It cannot change top-1 accuracy at all, and it
  should collapse the ECE reported above. This is the highest value-per-line
  change in the repo.
- **Augmented feature caching.** Cache `n` augmented views per training image
  instead of one clean view (`build_cache_loader` would need an `n_views`
  argument, and `features.extract_features` would stack them). That recovers
  augmentation for the probe while keeping most of the speed-up.
- **Partial fine-tuning.** Unfreeze only `layer4` plus the head. It sits between
  the two regimes implemented here and is usually the best accuracy-per-FLOP
  point on small datasets. `TransferModel.freeze_backbone` is the place to
  generalise into a per-block freeze.
- **Test-time augmentation.** Average logits over a horizontal flip and a couple
  of crops in `inference.collect_predictions`.
- **Attack the named weak classes.** `reports/<run>/per_class_test.csv` and the
  weakest-class figure name the specific flowers that fail. Class-balanced loss
  weights, or simply looking at those images, is a more targeted next step than
  a bigger backbone.
- **A new backbone** is a single registry entry in `models/backbones.py` plus a
  name in `SUPPORTED_BACKBONES`; add its last spatial block to `_TARGET_LAYERS`
  in `interpret/gradcam.py` and Grad-CAM keeps working.

## 9. References

- M-E. Nilsback and A. Zisserman. *Automated Flower Classification over a Large
  Number of Classes.* ICVGIP, 2008. (Oxford Flowers-102 dataset.)
- K. He et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
  (ResNet.)
- M. Tan and Q. Le. *EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks.* ICML, 2019.
- R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization.* ICCV, 2017.
- C. Guo et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
  (ECE, reliability diagrams, temperature scaling.)
- I. Loshchilov and F. Hutter. *Decoupled Weight Decay Regularization.* ICLR,
  2019. (AdamW.)
- PyTorch / torchvision, scikit-learn (metrics), matplotlib (figures).
