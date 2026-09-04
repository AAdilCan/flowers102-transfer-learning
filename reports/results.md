# Results

Oxford Flowers-102, held-out **test** split (6,149 images), trained on the
official 1,020-image train split with 1,020 images used for validation and
model selection. Confidence intervals are 2,000-sample percentile bootstraps
over the test set.

| run | top-1 | 95% CI | top-5 | balanced acc | macro-F1 | ECE |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| resnet18_probe | 0.8310 | [0.8216, 0.8401] | 0.9550 | 0.8511 | 0.8269 | 0.3303 |
| resnet50_probe | 0.8528 | [0.8440, 0.8621] | 0.9632 | 0.8640 | 0.8509 | 0.3188 |
| efficientnet_b0_probe | 0.8697 | [0.8611, 0.8784] | 0.9645 | 0.8831 | 0.8631 | 0.3415 |
| resnet18_finetune | 0.8413 | [0.8314, 0.8504] | 0.9489 | 0.8687 | 0.8412 | 0.3627 |

### Cost and training detail

| run | trainable params | epochs run | best epoch | best val acc | train time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| resnet18_probe | 52,326 | 40 | 32 | 0.8627 | reused |
| resnet50_probe | 208,998 | 36 | 26 | 0.8686 | reused |
| efficientnet_b0_probe | 130,662 | 39 | 29 | 0.8990 | reused |
| resnet18_finetune | 11,228,838 | 12 | 11 | 0.8735 | 1510 |

### Weakest classes (best run)

Lowest-recall classes for `efficientnet_b0_probe`:

| class | support | recall | precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| sweet pea | 36 | 0.528 | 0.452 | 0.487 |
| sword lily | 110 | 0.545 | 0.909 | 0.682 |
| canterbury bells | 20 | 0.550 | 0.379 | 0.449 |
| petunia | 238 | 0.592 | 0.865 | 0.703 |
| camellia | 71 | 0.620 | 0.786 | 0.693 |
| sweet william | 65 | 0.708 | 0.885 | 0.786 |
| corn poppy | 21 | 0.714 | 0.833 | 0.769 |
| snapdragon | 67 | 0.716 | 0.828 | 0.768 |
| hibiscus | 111 | 0.721 | 0.640 | 0.678 |
| balloon flower | 29 | 0.724 | 0.778 | 0.750 |

Figures for each run are in `reports/<run>/`.
