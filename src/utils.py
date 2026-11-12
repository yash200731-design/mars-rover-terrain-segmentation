"""
Utility Functions for Mars Rover Terrain Segmentation
AI4Mars Dataset (NASA JPL) - PyTorch Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Optional
import torch

# Four core traversability classes defined by NASA JPL AI4Mars
TERRAIN_CLASSES = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
}

# Ignore index for unannotated, out-of-range (>30m), or rover chassis pixels
IGNORE_INDEX = 255

# High-contrast, scientifically clear color palette for Martian surface analysis
# Soil -> Emerald Green (high traction, safe path)
# Bedrock -> Electric Blue (solid stone, firm traction)
# Sand -> Warning Yellow/Amber (slip/entrapment hazard)
# Big Rock -> Crimson Red (collision obstacle)
# Unlabeled -> Dark Gray/Black (ignored/null)
COLOR_MAP = {
    0: (46, 204, 113),   # Green
    1: (52, 152, 219),   # Blue
    2: (241, 196, 15),   # Yellow
    3: (231, 76, 60),    # Red
    IGNORE_INDEX: (30, 30, 30),  # Dark Charcoal
}

# Normalized RGB colors (0.0 to 1.0) for Matplotlib
COLOR_MAP_NORMALIZED = {
    k: tuple(c / 255.0 for c in v) for k, v in COLOR_MAP.items()
}

# ImageNet normalization parameters used for ResNet-34 encoder
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_segmentation_mask(mask: np.ndarray, color_map: Dict[int, Tuple[int, int, int]] = COLOR_MAP) -> np.ndarray:
    """
    Converts a 2D integer class mask (H, W) into an RGB image (H, W, 3)
    using the specified color palette.
    """
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        match_idx = (mask == class_id)
        rgb_mask[match_idx] = color
    return rgb_mask


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color_map: Dict[int, Tuple[int, int, int]] = COLOR_MAP,
    ignore_index: int = IGNORE_INDEX
) -> np.ndarray:
    """
    Blends an RGB image with a colored segmentation mask using alpha transparency.
    Ignored/unlabeled pixels retain the original image appearance without tint.
    """
    if image.dtype != np.uint8:
        # If float normalized, convert to uint8 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    rgb_mask = decode_segmentation_mask(mask, color_map=color_map)

    # Where mask is valid, perform alpha blend; where mask is ignored, keep original image
    valid_mask = (mask != ignore_index)[:, :, np.newaxis]
    overlay = np.where(valid_mask, (1 - alpha) * image + alpha * rgb_mask, image)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch image tensor (C, H, W) back to an RGB NumPy array (H, W, 3).
    """
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4,
    ignore_index: int = IGNORE_INDEX
) -> Dict[str, any]:
    """
    Calculates Pixel Accuracy, Per-Class IoU, Mean IoU (mIoU),
    Per-Class Dice (F1), Mean Dice, and the Confusion Matrix.
    
    Pixels equal to ignore_index are strictly excluded from all calculations.
    """
    # Flatten arrays
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Filter out ignored pixels
    valid_idx = (targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < num_classes)
    preds_valid = preds_flat[valid_idx]
    targets_valid = targets_flat[valid_idx]

    if len(targets_valid) == 0:
        return {
            "pixel_accuracy": 0.0,
            "iou_per_class": {cls_id: 0.0 for cls_id in range(num_classes)},
            "miou": 0.0,
            "dice_per_class": {cls_id: 0.0 for cls_id in range(num_classes)},
            "mean_dice": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64),
        }

    # Pixel Accuracy
    correct = np.sum(preds_valid == targets_valid)
    total_pixels = len(targets_valid)
    pixel_accuracy = float(correct / total_pixels)

    # Compute 4x4 Confusion Matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets_valid, preds_valid):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion_matrix[t, p] += 1

    # IoU and Dice per class
    iou_per_class = {}
    dice_per_class = {}
    ious = []
    dices = []

    for cls_id in range(num_classes):
        tp = confusion_matrix[cls_id, cls_id]
        fn = np.sum(confusion_matrix[cls_id, :]) - tp
        fp = np.sum(confusion_matrix[:, cls_id]) - tp

        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn

        iou = float(tp / denom_iou) if denom_iou > 0 else float("nan")
        dice = float((2 * tp) / denom_dice) if denom_dice > 0 else float("nan")

        iou_per_class[cls_id] = iou
        dice_per_class[cls_id] = dice

        if not np.isnan(iou):
            ious.append(iou)
        if not np.isnan(dice):
            dices.append(dice)

    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "iou_per_class": iou_per_class,
        "miou": miou,
        "dice_per_class": dice_per_class,
        "mean_dice": mean_dice,
        "confusion_matrix": confusion_matrix,
    }


def plot_sample(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    overlay: Optional[np.ndarray] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Renders an informative side-by-side visualization with a unified terrain legend.
    """
    num_cols = 1 + (ground_truth is not None) + (prediction is not None) + (overlay is not None)
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5), dpi=100)
    if num_cols == 1:
        axes = [axes]

    idx = 0
    # 1. Original RGB
    axes[idx].imshow(image)
    axes[idx].set_title("Original Mars Image", fontsize=12, fontweight="bold")
    axes[idx].axis("off")
    idx += 1

    # 2. Ground Truth
    if ground_truth is not None:
        gt_rgb = decode_segmentation_mask(ground_truth)
        axes[idx].imshow(gt_rgb)
        axes[idx].set_title("Ground-Truth Mask", fontsize=12, fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    # 3. Prediction
    if prediction is not None:
        pred_rgb = decode_segmentation_mask(prediction)
        axes[idx].imshow(pred_rgb)
        axes[idx].set_title("Predicted Mask", fontsize=12, fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    # 4. Overlay
    if overlay is not None:
        axes[idx].imshow(overlay)
        axes[idx].set_title("Prediction Overlay", fontsize=12, fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=COLOR_MAP_NORMALIZED[0], label="Soil (Safe/Tractable)"),
        mpatches.Patch(color=COLOR_MAP_NORMALIZED[1], label="Bedrock (Firm Traction)"),
        mpatches.Patch(color=COLOR_MAP_NORMALIZED[2], label="Sand (Slippage Hazard)"),
        mpatches.Patch(color=COLOR_MAP_NORMALIZED[3], label="Big Rock (Collision Hazard)"),
        mpatches.Patch(color=COLOR_MAP_NORMALIZED[IGNORE_INDEX], label="Unlabeled/Ignored"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=10,
        frameon=True,
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
