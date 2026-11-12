"""
Evaluation and Quantitative Assessment Script
Mars Rover Terrain Segmentation (NASA JPL AI4Mars)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ensure local src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import match_image_mask_pairs, MarsTerrainDataset, get_transforms
from src.model import build_unet_resnet34
from src.utils import (
    TERRAIN_CLASSES,
    IGNORE_INDEX,
    COLOR_MAP,
    calculate_metrics,
    decode_segmentation_mask,
    create_overlay,
    denormalize_image,
)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Generates and saves a normalized confusion matrix heatmap."""
    # Row-normalize to show recall per class
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    cax = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="bold")

    # Annotate cells with normalized percentage and raw counts
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = norm_cm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val * 100:.1f}%\n({count:,})",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_title("Normalized Confusion Matrix (Excl. Ignored Pixels)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[*] Confusion Matrix saved to: {save_path}")


def plot_prediction_grid(
    images: list,
    gts: list,
    preds: list,
    save_path: str,
    max_samples: int = 4
):
    """Renders a grid comparing Original Images, Ground Truth, Predictions, and Overlays."""
    num_samples = min(len(images), max_samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(18, 4.2 * num_samples), dpi=120)
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = [
        "1. Raw Mars Rover Image",
        "2. Ground Truth Mask",
        "3. Predicted Segmentation Mask",
        "4. Terrain Hazard Overlay",
    ]

    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(num_samples):
        img = images[i]
        gt = gts[i]
        pred = preds[i]
        overlay = create_overlay(img, pred, alpha=0.55)

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(decode_segmentation_mask(gt))
        axes[i, 1].axis("off")

        axes[i, 2].imshow(decode_segmentation_mask(pred))
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[*] Qualitative sample grid saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mars Rover Terrain Segmentation Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mars_terrain_model.pth",
        help="Path to trained model checkpoint .pth",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"D:\mars\ai4mars-dataset-merged-0.6",
        help="Root directory of AI4Mars dataset",
    )
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save evaluation plots")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--sample-limit", type=int, default=200, help="Number of validation images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for matching train/val split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    base_data = Path(args.data_dir)
    images_dir = base_data / "msl" / "mcam" / "images"
    labels_dir = base_data / "msl" / "mcam" / "labels" / "train"

    matched_images, matched_masks, _, _ = match_image_mask_pairs(
        images_dir=images_dir,
        labels_dir=labels_dir,
        verbose=False,
    )

    # Use same split as training
    _, val_imgs, _, val_masks = train_test_split(
        matched_images, matched_masks, test_size=0.20, random_state=args.seed, shuffle=True
    )

    if args.sample_limit and args.sample_limit < len(val_imgs):
        val_imgs = val_imgs[:args.sample_limit]
        val_masks = val_masks[:args.sample_limit]

    print(f"[*] Evaluating on {len(val_imgs)} validation images...")

    val_tf = get_transforms(is_train=False, target_size=(256, 256))
    val_dataset = MarsTerrainDataset(val_imgs, val_masks, transform=val_tf, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model & load weights
    model = build_unet_resnet34(num_classes=4, pretrained=False).to(device)

    chk_path = Path(args.checkpoint)
    if not chk_path.exists():
        print(f"[!] Warning: Checkpoint not found at {chk_path}.")
        print("[!] Generating architectural and pipeline validation test...")
    else:
        print(f"[*] Loading checkpoint weights from: {chk_path}")
        checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model.eval()

    all_preds = []
    all_targets = []
    sample_images = []
    sample_gts = []
    sample_preds = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            all_preds.append(preds_np)
            all_targets.append(masks_np)

            # Store first few images for visual qualitative grid
            if len(sample_images) < 4:
                for b in range(images.size(0)):
                    if len(sample_images) < 4:
                        sample_images.append(denormalize_image(images[b]))
                        sample_gts.append(masks_np[b])
                        sample_preds.append(preds_np[b])

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    metrics = calculate_metrics(
        predictions=all_preds,
        targets=all_targets,
        num_classes=4,
        ignore_index=IGNORE_INDEX,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("NASA AI4Mars Terrain Segmentation Quantitative Evaluation")
    print("=" * 65)
    print(f"Overall Pixel Accuracy: {metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"Mean Intersection over Union (mIoU): {metrics['miou'] * 100:.2f}%")
    print(f"Mean Dice Score (F1): {metrics['mean_dice'] * 100:.2f}%")
    print("-" * 65)
    print(f"{'Class ID':<10} {'Terrain Class':<15} {'IoU (%)':<15} {'Dice / F1 (%)':<15}")
    print("-" * 65)
    for cls_id, cls_name in TERRAIN_CLASSES.items():
        iou = metrics["iou_per_class"].get(cls_id, 0.0) * 100
        dice = metrics["dice_per_class"].get(cls_id, 0.0) * 100
        print(f"{cls_id:<10} {cls_name:<15} {iou:<15.2f} {dice:<15.2f}")
    print("=" * 65)

    # Save metrics JSON
    metrics_export = {
        "pixel_accuracy": metrics["pixel_accuracy"],
        "miou": metrics["miou"],
        "mean_dice": metrics["mean_dice"],
        "per_class_iou": {TERRAIN_CLASSES[k]: v for k, v in metrics["iou_per_class"].items()},
        "per_class_dice": {TERRAIN_CLASSES[k]: v for k, v in metrics["dice_per_class"].items()},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=4)

    # Plot Confusion Matrix
    class_names = [TERRAIN_CLASSES[i] for i in range(4)]
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=class_names,
        save_path=str(output_dir / "confusion_matrix.png"),
    )

    # Plot sample qualitative predictions
    if sample_images:
        plot_prediction_grid(
            images=sample_images,
            gts=sample_gts,
            preds=sample_preds,
            save_path=str(output_dir / "sample_predictions.png"),
            max_samples=4,
        )

    print(f"[*] All evaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
