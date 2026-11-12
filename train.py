"""
Training Script for Mars Rover Terrain Segmentation
AI4Mars Dataset (NASA JPL) - PyTorch U-Net (ResNet-34)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ensure local src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import match_image_mask_pairs, MarsTerrainDataset, get_transforms
from src.model import build_unet_resnet34
from src.utils import TERRAIN_CLASSES, IGNORE_INDEX, calculate_metrics


def set_seed(seed: int = 42):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    start_time = time.time()
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [B, 4, H, W]
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch+1:02d}/{num_epochs:02d}] "
                f"Batch [{batch_idx+1:03d}/{total_batches:03d}] "
                f"Train Loss: {loss.item():.4f} "
                f"({elapsed:.1f}s)"
            )

    epoch_loss = running_loss / total_batches
    return epoch_loss


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = masks.cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets)

    val_loss = running_loss / max(1, len(dataloader))
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = calculate_metrics(
        predictions=all_preds,
        targets=all_targets,
        num_classes=num_classes,
        ignore_index=IGNORE_INDEX,
    )
    metrics["val_loss"] = val_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Mars Rover Terrain Segmentation Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"D:\mars\ai4mars-dataset-merged-0.6",
        help="Root directory containing AI4Mars dataset",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for model and logs")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional subset size for fast testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2, help="DataLoader worker processes")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Resolve paths for MSL MastCam dataset
    base_data = Path(args.data_dir)
    images_dir = base_data / "msl" / "mcam" / "images"
    labels_dir = base_data / "msl" / "mcam" / "labels" / "train"

    matched_images, matched_masks, unmatched_imgs, unmatched_lbls = match_image_mask_pairs(
        images_dir=images_dir,
        labels_dir=labels_dir,
        verbose=True,
    )

    if len(matched_images) == 0:
        raise RuntimeError("No matching image-mask pairs found. Please verify dataset paths.")

    if args.sample_limit and args.sample_limit < len(matched_images):
        print(f"[*] Limiting dataset to first {args.sample_limit} pairs for quick trial.")
        matched_images = matched_images[:args.sample_limit]
        matched_masks = matched_masks[:args.sample_limit]

    # 80/20 Train-Validation Split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        matched_images, matched_masks, test_size=0.20, random_state=args.seed, shuffle=True
    )

    print(f"[*] Training Samples:   {len(train_imgs)}")
    print(f"[*] Validation Samples: {len(val_imgs)}")

    # Transforms & Datasets
    train_tf = get_transforms(is_train=True, target_size=(256, 256))
    val_tf = get_transforms(is_train=False, target_size=(256, 256))

    train_dataset = MarsTerrainDataset(train_imgs, train_masks, transform=train_tf, is_train=True)
    val_dataset = MarsTerrainDataset(val_imgs, val_masks, transform=val_tf, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize U-Net ResNet-34 Architecture
    model = build_unet_resnet34(num_classes=4, pretrained=True).to(device)

    # Loss function with Ignore Index
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save class configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "architecture": "U-Net",
                "encoder": "resnet34",
                "classes": TERRAIN_CLASSES,
                "ignore_index": IGNORE_INDEX,
                "input_resolution": [256, 256],
                "train_samples": len(train_imgs),
                "val_samples": len(val_imgs),
                "batch_size": args.batch_size,
                "initial_lr": args.lr,
            },
            f,
            indent=4,
        )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_accuracy": []}

    print("\n" + "=" * 60)
    print("Beginning Training: U-Net (ResNet-34) on AI4Mars MastCam")
    print("=" * 60)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
        )

        metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=4,
        )

        val_loss = metrics["val_loss"]
        miou = metrics["miou"]
        acc = metrics["pixel_accuracy"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(miou)
        history["val_accuracy"].append(acc)

        print("-" * 60)
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] Summary:")
        print(f"  Train Loss:     {train_loss:.4f}")
        print(f"  Val Loss:       {val_loss:.4f}")
        print(f"  Val Pixel Acc:  {acc * 100:.2f}%")
        print(f"  Val Mean IoU:   {miou * 100:.2f}%")
        for cls_id, cls_name in TERRAIN_CLASSES.items():
            cls_iou = metrics["iou_per_class"].get(cls_id, 0.0) * 100
            cls_dice = metrics["dice_per_class"].get(cls_id, 0.0) * 100
            print(f"    - {cls_name:<10}: IoU = {cls_iou:5.2f}% | Dice = {cls_dice:5.2f}%")
        print("-" * 60)

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "mars_terrain_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_miou": miou,
                    "metrics": metrics,
                    "classes": TERRAIN_CLASSES,
                },
                best_model_path,
            )
            print(f"[Checkpoint] Best model saved to: {best_model_path} (Val Loss: {val_loss:.4f})")

    # Save final history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Plot training loss & mIoU curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="#E27B58", lw=2)
    plt.plot(history["val_loss"], label="Val Loss", color="#3498DB", lw=2)
    plt.title("Cross-Entropy Loss (Ignore Index 255)", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([m * 100 for m in history["val_miou"]], label="Val mIoU (%)", color="#2ECC71", lw=2)
    plt.plot([a * 100 for a in history["val_accuracy"]], label="Val Pixel Acc (%)", color="#F1C40F", lw=2)
    plt.title("Validation Performance Metrics", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    curve_path = output_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[*] Training curves saved to: {curve_path}")
    print("[*] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
