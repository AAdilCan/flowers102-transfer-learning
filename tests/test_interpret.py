"""Tests for Grad-CAM and its rendering.

The interesting case is the linear-probe model: its backbone parameters all
have ``requires_grad=False``, which is exactly the setup where a naive
implementation raises "element 0 of tensors does not require grad".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from imgclf.config import Config
from imgclf.interpret import (
    GradCAM,
    denormalize,
    explain,
    overlay_cam,
    resolve_target_layer,
    save_gradcam_panel,
)
from imgclf.models import build_model

IMAGE_SIZE = 64


def _model(backbone: str = "resnet18", mode: str = "linear_probe"):
    cfg = Config.from_dict(
        {
            "model": {"backbone": backbone, "mode": mode, "num_classes": 10},
            "data": {"image_size": IMAGE_SIZE},
        }
    )
    return build_model(cfg, pretrained=False)


def _image() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_resolve_target_layer_uses_the_registry() -> None:
    model = _model("resnet18")
    assert resolve_target_layer(model, "resnet18") is model.backbone.layer4


def test_resolve_target_layer_falls_back_to_last_conv() -> None:
    model = _model("resnet18")
    layer = resolve_target_layer(model, backbone_name=None)
    assert isinstance(layer, nn.Conv2d)


def test_resolve_target_layer_rejects_conv_free_backbone() -> None:
    model = _model("resnet18")
    model.backbone = nn.Sequential(nn.Flatten(), nn.Linear(4, 4))
    with pytest.raises(ValueError, match="no Conv2d"):
        resolve_target_layer(model, backbone_name=None)


@pytest.mark.parametrize("mode", ["linear_probe", "finetune"])
def test_cam_shape_and_range(mode: str) -> None:
    """A frozen backbone must still produce a heatmap, not a grad error."""
    cam, label, prob = explain(_model(mode=mode), _image(), backbone_name="resnet18")
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0 + 1e-6
    assert 0 <= label < 10
    assert 0.0 <= prob <= 1.0


def test_explaining_a_chosen_class_reports_that_class() -> None:
    cam, label, prob = explain(_model(), _image(), backbone_name="resnet18", class_idx=7)
    assert label == 7
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_different_classes_give_different_heatmaps() -> None:
    model = _model()
    image = _image()
    layer = resolve_target_layer(model, "resnet18")
    with GradCAM(model, layer) as gradcam:
        cam_a, _, _ = gradcam(image, class_idx=0)
        cam_b, _, _ = gradcam(image, class_idx=5)
    assert not torch.allclose(cam_a, cam_b)


def test_hooks_are_released_on_exit() -> None:
    model = _model()
    layer = resolve_target_layer(model, "resnet18")
    with GradCAM(model, layer):
        assert len(layer._forward_hooks) == 1
    assert len(layer._forward_hooks) == 0
    assert len(layer._backward_hooks) + len(layer._forward_pre_hooks) == 0


def test_training_mode_is_restored() -> None:
    model = _model(mode="finetune")
    model.train()
    explain(model, _image(), backbone_name="resnet18")
    assert model.training is True

    model.eval()
    explain(model, _image(), backbone_name="resnet18")
    assert model.training is False


def test_model_weights_are_not_updated_by_explaining() -> None:
    """Grad-CAM backpropagates but must never step the optimiser."""
    model = _model(mode="finetune")
    before = model.head.fc.weight.detach().clone()
    explain(model, _image(), backbone_name="resnet18")
    assert torch.equal(before, model.head.fc.weight.detach())


def test_batch_input_is_rejected() -> None:
    model = _model()
    layer = resolve_target_layer(model, "resnet18")
    with GradCAM(model, layer) as gradcam:
        with pytest.raises(ValueError, match="single image"):
            gradcam(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        with pytest.raises(ValueError, match="single image"):
            gradcam(torch.randn(3, IMAGE_SIZE, IMAGE_SIZE))


def test_denormalize_inverts_the_normalisation() -> None:
    from imgclf.config import IMAGENET_MEAN, IMAGENET_STD
    from torchvision import transforms

    raw = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
    normalised = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(raw)
    restored = denormalize(normalised.unsqueeze(0))
    assert restored.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert np.allclose(restored, raw.permute(1, 2, 0).numpy(), atol=1e-5)


def test_denormalize_rejects_a_batch() -> None:
    with pytest.raises(ValueError, match="single image"):
        denormalize(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))


def test_overlay_matches_image_shape() -> None:
    image = _image()
    cam = torch.rand(IMAGE_SIZE, IMAGE_SIZE)
    blended = overlay_cam(image, cam)
    assert blended.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert blended.min() >= 0.0 and blended.max() <= 1.0


def test_panel_is_written(tmp_path) -> None:
    images = [_image(), _image()]
    cams = [torch.rand(IMAGE_SIZE, IMAGE_SIZE) for _ in images]
    path = save_gradcam_panel(images, cams, ["a", "b"], tmp_path / "panel.png")
    assert path.exists() and path.stat().st_size > 0


def test_panel_validates_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="same length"):
        save_gradcam_panel([_image()], [], ["a"], tmp_path / "p.png")
    with pytest.raises(ValueError, match="nothing to plot"):
        save_gradcam_panel([], [], [], tmp_path / "p.png")
