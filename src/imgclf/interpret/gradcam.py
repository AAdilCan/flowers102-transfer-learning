"""Grad-CAM for the transfer models.

Grad-CAM weights the last convolutional feature maps by the gradient of a class
score with respect to those maps, giving a coarse heatmap of the pixels that
pushed the prediction. On Flowers-102 it answers the question a top-1 number
cannot: is the model looking at the flower, or at the background it happens to
co-occur with?

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""

from __future__ import annotations

from types import TracebackType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging_utils import get_logger
from ..models import TransferModel

logger = get_logger("gradcam")

# Where the last spatial feature map lives in each supported backbone. Falling
# back to "the last Conv2d in the tree" also works, but naming the block keeps
# the hook on the residual/MBConv output rather than on an internal 1x1 conv.
_TARGET_LAYERS: dict[str, str] = {
    "resnet18": "layer4",
    "resnet50": "layer4",
    "efficientnet_b0": "features",
}


def resolve_target_layer(model: TransferModel, backbone_name: str | None = None) -> nn.Module:
    """Pick the layer whose activations Grad-CAM hooks.

    Uses the per-backbone registry when the architecture is known, otherwise
    falls back to the last ``Conv2d`` in the backbone.
    """
    backbone = model.backbone
    if backbone_name and backbone_name in _TARGET_LAYERS:
        module = getattr(backbone, _TARGET_LAYERS[backbone_name], None)
        if module is not None:
            return module

    convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("backbone contains no Conv2d layer to attach Grad-CAM to")
    logger.debug("falling back to the last Conv2d as the Grad-CAM target")
    return convs[-1]


class GradCAM:
    """Compute Grad-CAM heatmaps for a :class:`TransferModel`.

    Hooks are registered on construction and released by :meth:`remove` (or by
    leaving the ``with`` block), so a long-lived model is never left with stray
    hooks after an explanation is produced.
    """

    def __init__(self, model: TransferModel, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def remove(self) -> None:
        """Detach the forward/backward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.remove()

    def __call__(
        self, image: torch.Tensor, class_idx: int | None = None
    ) -> tuple[torch.Tensor, int, float]:
        """Return ``(cam, class_idx, probability)`` for one image.

        ``image`` is a normalised ``(1, 3, H, W)`` tensor. ``cam`` comes back as
        an ``(H, W)`` tensor in ``[0, 1]``, already upsampled to the input size.
        When ``class_idx`` is omitted the model's own top-1 class is explained.
        """
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError(f"expected a single image (1, 3, H, W), got {tuple(image.shape)}")

        was_training = self.model.training
        self.model.eval()

        # A linear-probe backbone has requires_grad=False on every parameter, so
        # nothing in the forward pass would build a graph and backward() would
        # fail. Making the *input* require grad is enough: the activations then
        # carry grad_fn even though no weight is being trained.
        image = image.clone().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(image)
            probs = logits.softmax(dim=1)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]

            self.model.zero_grad(set_to_none=True)
            if image.grad is not None:
                image.grad = None
            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks captured nothing; is the target layer used?")

        activations = self._activations.detach()[0]        # (C, h, w)
        gradients = self._gradients.detach()[0]            # (C, h, w)

        # Channel weights are the spatially averaged gradients; the CAM is the
        # positive part of their weighted sum over channels.
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=0))

        cam = F.interpolate(
            cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        # Normalise to [0, 1]; a fully-zero CAM (no positive evidence) stays zero.
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        self.model.train(was_training)
        return cam.cpu(), int(class_idx), float(probs[0, class_idx].item())


def explain(
    model: TransferModel,
    image: torch.Tensor,
    *,
    backbone_name: str | None = None,
    class_idx: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    """One-shot Grad-CAM that resolves the target layer and cleans up its hooks."""
    layer = resolve_target_layer(model, backbone_name)
    with GradCAM(model, layer) as cam:
        return cam(image, class_idx)
