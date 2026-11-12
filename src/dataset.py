"""
Dataset and Preprocessing Pipeline for Mars Rover Terrain Segmentation
NASA JPL AI4Mars Dataset
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def match_image_mask_pairs(
    images_dir: str or Path,
    labels_dir: str or Path,
    verbose: bool = True
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Automatically associates raw Mars rover images with their corresponding segmentation masks.
    
    Handles known naming conventions:
    - Base MastCam images: '<stem>.JPG' or '<stem>.jpg'
    - MastCam label masks: '<stem>_15033_merged.png'
    - General merged labels: '<stem>_merged.png' or '<stem>_merged<N>.png'
    
    Returns:
        matched_images: List of full paths to matched images
        matched_masks: List of full paths to matched masks
        unmatched_images: List of unmatched image filenames
        unmatched_masks: List of unmatched mask filenames
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Gather images (case-insensitive extension check)
    valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in valid_img_exts
    ]

    # Gather masks
    valid_mask_exts = {".png", ".bmp"}
    mask_files = [
        f for f in os.listdir(labels_dir)
        if Path(f).suffix.lower() in valid_mask_exts
    ]

    # Build lookup table for masks: mapping cleaned base stem -> filename
    # E.g. '1150ML0052000010501746E01_XXXX_15033_merged.png' -> '1150ML0052000010501746E01_XXXX'
    mask_lookup: Dict[str, str] = {}
    for mf in mask_files:
        stem = Path(mf).stem
        # Regex to strip _<workflow_id>_merged or _merged<count> suffixes
        cleaned_stem = re.sub(r'(_\d+)?_merged(\d+)?$', '', stem)
        mask_lookup[cleaned_stem] = mf

    matched_images: List[str] = []
    matched_masks: List[str] = []
    unmatched_images: List[str] = []

    matched_mask_filenames = set()

    for imf in sorted(img_files):
        img_stem = Path(imf).stem
        if img_stem in mask_lookup:
            mf = mask_lookup[img_stem]
            matched_images.append(str(images_dir / imf))
            matched_masks.append(str(labels_dir / mf))
            matched_mask_filenames.add(mf)
        else:
            unmatched_images.append(imf)

    unmatched_masks = [mf for mf in mask_files if mf not in matched_mask_filenames]

    if verbose:
        print("=" * 60)
        print("NASA AI4Mars Dataset Pair Matching Report")
        print("=" * 60)
        print(f"Total raw images detected: {len(img_files)}")
        print(f"Total label masks detected: {len(mask_files)}")
        print(f"Successfully matched pairs: {len(matched_images)}")
        print(f"Unmatched images: {len(unmatched_images)}")
        print(f"Unmatched masks: {len(unmatched_masks)}")
        print("=" * 60)

    return matched_images, matched_masks, unmatched_images, unmatched_masks


class MarsTerrainDataset(Dataset):
    """
    PyTorch Dataset for NASA AI4Mars Semantic Terrain Segmentation.
    
    Classes:
        0: Soil
        1: Bedrock
        2: Sand
        3: Big Rock
        255: Unlabeled / Null / Ignored (mapped from raw 4 or 255)
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (256, 256),
        is_train: bool = False,
    ):
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch in count: {len(image_paths)} images vs {len(mask_paths)} masks."
            )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size  # (width, height)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def _remap_mask_values(self, mask: np.ndarray) -> np.ndarray:
        """
        Standardizes mask values to [0, 1, 2, 3] and IGNORE_INDEX (255).
        Raw AI4Mars MSL Mastcam uses:
          0: soil, 1: bedrock, 2: sand, 3: big rock, 4: null/unlabeled.
        Raw AI4Mars NavCam uses:
          0: soil, 1: bedrock, 2: sand, 3: big rock, 255: null/unlabeled.
        Any label value > 3 (including 4 and 255) is mapped to IGNORE_INDEX (255).
        """
        # Create a copy with uint8 type
        processed = mask.copy()
        # Anything not in valid classes {0, 1, 2, 3} becomes ignore_index (255)
        invalid_mask = (processed < 0) | (processed > 3)
        processed[invalid_mask] = IGNORE_INDEX
        return processed.astype(np.uint8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            # 1. Load Image (RGB)
            with Image.open(img_path) as pil_img:
                image = pil_img.convert("RGB")
                if image.size != self.target_size:
                    image = image.resize(self.target_size, resample=Image.BILINEAR)
                image_np = np.array(image, dtype=np.uint8)

            # 2. Load Mask (Grayscale / 'L' mode)
            with Image.open(mask_path) as pil_mask:
                # Nearest neighbor resizing is critical for segmentation labels
                # to strictly preserve class IDs without interpolation artifacts!
                if pil_mask.size != self.target_size:
                    pil_mask = pil_mask.resize(self.target_size, resample=Image.NEAREST)
                mask_np = np.array(pil_mask, dtype=np.uint8)

            # 3. Standardize class IDs
            mask_np = self._remap_mask_values(mask_np)

            # 4. Apply transforms / data augmentation
            if self.transform is not None:
                if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
                    augmented = self.transform(image=image_np, mask=mask_np)
                    image_tensor = augmented["image"]
                    mask_tensor = augmented["mask"]
                    if not isinstance(image_tensor, torch.Tensor):
                        image_tensor = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float()
                    if not isinstance(mask_tensor, torch.Tensor):
                        mask_tensor = torch.from_numpy(mask_tensor).long()
                    else:
                        mask_tensor = mask_tensor.long()
                    return image_tensor, mask_tensor
                else:
                    # Custom callable transform
                    image_np, mask_np = self.transform(image_np, mask_np)

            # Fallback tensor conversion & ImageNet normalization
            img_normalized = (image_np / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            image_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_np).long()

            return image_tensor, mask_tensor

        except Exception as e:
            print(f"Warning: Corrupted sample at index {idx} ({img_path}): {e}")
            # Safe recovery: return next item or zeroes
            fallback_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(fallback_idx)


class IdentityTransform:
    """Top-level picklable callable transform fallback."""
    def __call__(self, img: np.ndarray, mask: np.ndarray):
        return img, mask


def get_transforms(is_train: bool = True, target_size: Tuple[int, int] = (256, 256)):
    """
    Creates Albumentations augmentation pipelines.
    - Training: Geometric & photometric augmentations preserving mask class values.
    - Validation/Testing: Deterministic resize and ImageNet normalization.
    """
    if not HAS_ALBUMENTATIONS:
        return IdentityTransform()

    if is_train:
        return A.Compose([
            A.Resize(target_size[1], target_size[0], interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=0,  # Constant border
                value=0,        # Black fill for image
                mask_value=IGNORE_INDEX,  # Ignore index for mask boundary
                p=0.4
            ),
            A.Normalize(
                mean=tuple(IMAGENET_MEAN),
                std=tuple(IMAGENET_STD),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(target_size[1], target_size[0], interpolation=1),
            A.Normalize(
                mean=tuple(IMAGENET_MEAN),
                std=tuple(IMAGENET_STD),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
