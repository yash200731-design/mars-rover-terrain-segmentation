"""
Mars Rover Terrain Segmentation Package
AI4Mars Dataset Semantic Segmentation using PyTorch & U-Net (ResNet-34)
"""

from .dataset import MarsTerrainDataset, get_transforms, match_image_mask_pairs
from .model import build_unet_resnet34
from .utils import (
    TERRAIN_CLASSES,
    COLOR_MAP,
    COLOR_MAP_NORMALIZED,
    decode_segmentation_mask,
    create_overlay,
    calculate_metrics,
    plot_sample,
)

__all__ = [
    "MarsTerrainDataset",
    "get_transforms",
    "match_image_mask_pairs",
    "build_unet_resnet34",
    "TERRAIN_CLASSES",
    "COLOR_MAP",
    "COLOR_MAP_NORMALIZED",
    "decode_segmentation_mask",
    "create_overlay",
    "calculate_metrics",
    "plot_sample",
]
