"""Interpretability: Grad-CAM heatmaps and their rendering."""

from .gradcam import GradCAM, explain, resolve_target_layer
from .visualize import denormalize, overlay_cam, save_gradcam_panel

__all__ = [
    "GradCAM",
    "explain",
    "resolve_target_layer",
    "denormalize",
    "overlay_cam",
    "save_gradcam_panel",
]
