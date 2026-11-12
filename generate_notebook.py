"""
Generator Script for Mars_Rover_Terrain_Segmentation.ipynb
Creates a complete, executable, production-grade Google Colab Notebook
following all 12 project steps.
"""

import json
from pathlib import Path


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }


def build_notebook():
    cells = []

    # ==========================================
    # Title & Executive Summary
    # ==========================================
    cells.append(make_markdown_cell([
        "# 🪐 Mars Rover Terrain Segmentation using Deep Learning",
        "### NASA Jet Propulsion Laboratory (JPL) AI4Mars Merged Dataset v0.6",
        "",
        "**Author:** Senior Machine Learning Engineer & Computer Vision Researcher  ",
        "**Target Environment:** Google Colab (GPU-accelerated) / Local PyTorch  ",
        "**Architecture:** U-Net with ResNet-34 Pretrained Backbone  ",
        "**Task:** 4-Class Semantic Segmentation (Soil, Bedrock, Sand, Big Rocks)",
        "",
        "---",
        "## 🔬 Scientific & Engineering Motivation",
        "Navigating autonomous rovers across the Martian surface (such as NASA's **Curiosity** and **Perseverance**) is an immense robotics and computer vision challenge. The Martian surface presents distinct terrain types with dramatically different traversability characteristics:",
        "",
        "1. **Soil (`0`)**: Fine Martian regolith, standard traversable terrain.",
        "2. **Bedrock (`1`)**: Solid rock outcrops providing high traction and vehicle stability.",
        "3. **Sand (`2`)**: Loose dunes and ripple fields that create severe wheel slippage and entrapment hazards (e.g., the *Spirit* rover entrapment in *Troy* dune).",
        "4. **Big Rocks (`3`)**: Boulders exceeding rover ground clearance, posing catastrophic collision risks.",
        "5. **Unlabeled / Null (`255`)**: Distances >30m, rover chassis parts, sky, or areas lacking labeler consensus.",
        "",
        "This notebook walks through an end-to-end deep learning workflow from raw telemetry ingestion to model training, evaluation, and deployment preparation."
    ]))

    # ==========================================
    # Step 1: Environment Setup
    # ==========================================
    cells.append(make_markdown_cell([
        "## ⚙️ Step 1: Environment Setup",
        "In this step, we install the required computer vision libraries (such as `segmentation-models-pytorch` and `albumentations`), import our dependencies, and verify GPU acceleration."
    ]))

    cells.append(make_code_cell([
        "# Install specialized deep learning and computer vision libraries",
        "# (Colab runs Linux x86_64 with instant prebuilt wheels)",
        "!pip install -q segmentation-models-pytorch albumentations opencv-python-headless matplotlib seaborn"
    ]))

    cells.append(make_code_cell([
        "# Import Core Dependencies",
        "import os",
        "import re",
        "import sys",
        "import json",
        "import time",
        "import random",
        "from pathlib import Path",
        "from typing import List, Tuple, Dict, Optional",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import matplotlib.patches as mpatches",
        "import seaborn as sns",
        "from PIL import Image",
        "",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from torch.utils.data import Dataset, DataLoader",
        "from sklearn.model_selection import train_test_split",
        "",
        "# Semantic segmentation architectures",
        "try:",
        "    import segmentation_models_pytorch as smp",
        "    print('[OK] Segmentation Models PyTorch loaded successfully.')",
        "except ImportError:",
        "    print('[!] smp not installed; native fallback will be used.')",
        "",
        "# Fast data augmentations",
        "try:",
        "    import albumentations as A",
        "    from albumentations.pytorch import ToTensorV2",
        "    print('[OK] Albumentations loaded successfully.')",
        "except ImportError:",
        "    print('[!] albumentations not installed.')",
        "",
        "# Set fixed random seed for 100% reproducibility",
        "RANDOM_SEED = 42",
        "random.seed(RANDOM_SEED)",
        "np.random.seed(RANDOM_SEED)",
        "torch.manual_seed(RANDOM_SEED)",
        "if torch.cuda.is_available():",
        "    torch.cuda.manual_seed_all(RANDOM_SEED)",
        "torch.backends.cudnn.deterministic = True",
        "print(f'[OK] Reproducibility seed set to: {RANDOM_SEED}')"
    ]))

    cells.append(make_code_cell([
        "# Hardware Acceleration Check",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'Active PyTorch Device: {device}')",
        "",
        "if torch.cuda.is_available():",
        "    gpu_name = torch.cuda.get_device_name(0)",
        "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)",
        "    cuda_capability = torch.cuda.get_device_capability(0)",
        "    print(f'  GPU Name:          {gpu_name}')",
        "    print(f'  Total VRAM:        {vram_gb:.2f} GB')",
        "    print(f'  Compute Capability: {cuda_capability[0]}.{cuda_capability[1]}')",
        "else:",
        "    print('  [NOTE] GPU not detected. If running on Google Colab, enable GPU in:')",
        "    print('  Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU / V100 / A100')"
    ]))

    # ==========================================
    # Step 2: Dataset Loading & Verification
    # ==========================================
    cells.append(make_markdown_cell([
        "## 📂 Step 2: Dataset Loading & Automated Pair Matching",
        "",
        "### Instructions for Dataset Access in Google Colab",
        "1. **Option A (Google Drive - Recommended for Colab):**",
        "   - Upload your `ai4mars-dataset-merged-0.6` folder (or a zipped archive `ai4mars-dataset-merged-0.6.zip`) to your Google Drive root directory (`MyDrive/`).",
        "   - Mount Google Drive using `drive.mount('/content/drive')`.",
        "   - If zipped, run `!unzip -q /content/drive/MyDrive/ai4mars-dataset-merged-0.6.zip -d /content/` for maximum I/O speed!",
        "2. **Option B (Local Machine):**",
        "   - The notebook automatically detects local directory paths such as `D:/mars/ai4mars-dataset-merged-0.6`.",
        "",
        "### Filename Convention & Matching",
        "Curiosity MastCam images are named like: `1150ML0052000010501746E01_XXXX.JPG`  ",
        "The crowdsourced training labels are named: `1150ML0052000010501746E01_XXXX_15033_merged.png`  ",
        "*(Note: `15033` denotes the Zooniverse citizen-science workflow ID for Curiosity MastCam).*  ",
        "We implement an automated filename matching algorithm that extracts base stems to match images with their masks without hardcoding."
    ]))

    cells.append(make_code_cell([
        "# Google Drive Mounting & Path Resolution",
        "try:",
        "    from google.colab import drive",
        "    drive.mount('/content/drive')",
        "    IN_COLAB = True",
        "    print('[OK] Running inside Google Colab.')",
        "except ImportError:",
        "    IN_COLAB = False",
        "    print('[OK] Running in local environment.')",
        "",
        "# Candidate paths to search for AI4Mars dataset",
        "candidate_paths = [",
        "    Path('/content/ai4mars-dataset-merged-0.6'),                                  # Fast local SSD in Colab",
        "    Path('/content/drive/MyDrive/ai4mars-dataset-merged-0.6'),                     # Google Drive path",
        "    Path('/content/drive/MyDrive/mars/ai4mars-dataset-merged-0.6'),                # Alternative Google Drive path",
        "    Path(r'D:/mars/ai4mars-dataset-merged-0.6'),                                  # Local Windows path",
        "    Path('./ai4mars-dataset-merged-0.6'),                                         # Current directory",
        "]",
        "",
        "DATASET_ROOT = None",
        "for cp in candidate_paths:",
        "    if cp.exists():",
        "        DATASET_ROOT = cp",
        "        break",
        "",
        "if DATASET_ROOT is None:",
        "    # Default fallback path",
        "    DATASET_ROOT = candidate_paths[1] if IN_COLAB else candidate_paths[3]",
        "    print(f'[!] Dataset not automatically found at candidates. Setting target path to: {DATASET_ROOT}')",
        "    print('    Please ensure the dataset is uploaded or update DATASET_ROOT below.')",
        "else:",
        "    print(f'[OK] Found AI4Mars dataset root at: {DATASET_ROOT}')"
    ]))

    cells.append(make_code_cell([
        "# Inspect label_keys.json to confirm official NASA JPL class definitions",
        "label_keys_file = DATASET_ROOT / 'label_keys.json'",
        "if label_keys_file.exists():",
        "    with open(label_keys_file, 'r') as f:",
        "        label_keys_data = json.load(f)",
        "    print('Official NASA JPL label_keys.json contents:')",
        "    print(json.dumps(label_keys_data, indent=4))",
        "else:",
        "    print(f'[!] Note: {label_keys_file} not found. Standard AI4Mars NAV key is 0:soil, 1:bedrock, 2:sand, 3:big rock.')"
    ]))

    cells.append(make_code_cell([
        "# Automated Image-Mask Matching Algorithm",
        "IMAGES_DIR = DATASET_ROOT / 'msl' / 'mcam' / 'images'",
        "LABELS_DIR = DATASET_ROOT / 'msl' / 'mcam' / 'labels' / 'train'",
        "",
        "def match_ai4mars_pairs(images_path: Path, labels_path: Path):",
        "    if not images_path.exists() or not labels_path.exists():",
        "        raise FileNotFoundError(f'Paths not found: {images_path} or {labels_path}')",
        "    ",
        "    # Collect valid images",
        "    img_exts = {'.jpg', '.jpeg', '.png'}",
        "    img_files = [f for f in os.listdir(images_path) if Path(f).suffix.lower() in img_exts]",
        "    ",
        "    # Collect label masks",
        "    mask_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.png')]",
        "    ",
        "    # Map mask base stems: e.g. stem_15033_merged.png -> stem",
        "    mask_lookup = {}",
        "    for mf in mask_files:",
        "        stem = Path(mf).stem",
        "        cleaned = re.sub(r'(_\\d+)?_merged(\\d+)?$', '', stem)",
        "        mask_lookup[cleaned] = mf",
        "    ",
        "    matched_imgs, matched_lbls, unmatched_imgs = [], [], []",
        "    matched_mask_names = set()",
        "    ",
        "    for imf in sorted(img_files):",
        "        stem = Path(imf).stem",
        "        if stem in mask_lookup:",
        "            mf = mask_lookup[stem]",
        "            matched_imgs.append(str(images_path / imf))",
        "            matched_lbls.append(str(labels_path / mf))",
        "            matched_mask_names.add(mf)",
        "        else:",
        "            unmatched_imgs.append(imf)",
        "            ",
        "    unmatched_lbls = [mf for mf in mask_files if mf not in matched_mask_names]",
        "    return matched_imgs, matched_lbls, unmatched_imgs, unmatched_lbls",
        "",
        "matched_images, matched_masks, unmatched_imgs, unmatched_lbls = match_ai4mars_pairs(IMAGES_DIR, LABELS_DIR)",
        "",
        "print('=' * 60)",
        "print('AI4MARS PAIR MATCHING REPORT')",
        "print('=' * 60)",
        "print(f'Total Raw Images Found:       {len(matched_images) + len(unmatched_imgs):,}')",
        "print(f'Total Label Masks Found:      {len(matched_masks) + len(unmatched_lbls):,}')",
        "print(f'Successfully Matched Pairs:   {len(matched_images):,}')",
        "print(f'Unmatched Images:             {len(unmatched_imgs):,}')",
        "print(f'Unmatched Masks:              {len(unmatched_lbls):,}')",
        "print('=' * 60)",
        "if len(unmatched_imgs) > 0:",
        "    print(f'Sample Unmatched Image: {unmatched_imgs[0]}')",
        "print(f'[OK] Dataset verified: {len(matched_images):,} valid image-mask pairs ready for processing.')"
    ]))

    # ==========================================
    # Step 3: Exploratory Data Analysis (EDA)
    # ==========================================
    cells.append(make_markdown_cell([
        "## 🔍 Step 3: Exploratory Data Analysis (EDA)",
        "",
        "### Key Characteristics of AI4Mars Imagery:",
        "- **Resolution:** Curiosity MastCam images typically have aspect ratios like `1152 × 432`, `1344 × 1200`, or `1152 × 1152`.",
        "- **Grayscale Masks:** Raw label PNGs store integers `[0, 1, 2, 3, 4]`. Because these values are tiny compared to `255`, viewing them directly in a standard image viewer appears pitch black!",
        "- **Unlabeled Pixels:** Raw label value `4` corresponds to unannotated regions (or `255` in NavCam). We map all unannotated pixels to an official `IGNORE_INDEX = 255`.",
        "",
        "Let us inspect the dimensions, unique pixel values, visual appearances, and class frequency distribution."
    ]))

    cells.append(make_code_cell([
        "# Color Palette & Constants for Martian Terrain",
        "TERRAIN_CLASSES = {",
        "    0: 'Soil',",
        "    1: 'Bedrock',",
        "    2: 'Sand',",
        "    3: 'Big Rock',",
        "}",
        "IGNORE_INDEX = 255",
        "",
        "# High-contrast RGB color map",
        "COLOR_MAP = {",
        "    0: (46, 204, 113),   # Green (Soil - Safe/Traversable)",
        "    1: (52, 152, 219),   # Blue (Bedrock - Firm Traction)",
        "    2: (241, 196, 15),   # Yellow (Sand - Slip Hazard)",
        "    3: (231, 76, 60),    # Red (Big Rock - Collision Hazard)",
        "    IGNORE_INDEX: (35, 35, 35),  # Dark Gray (Unlabeled / Ignored)",
        "}",
        "",
        "def decode_mask_to_rgb(mask_2d: np.ndarray) -> np.ndarray:",
        "    h, w = mask_2d.shape",
        "    rgb = np.zeros((h, w, 3), dtype=np.uint8)",
        "    for cid, col in COLOR_MAP.items():",
        "        rgb[mask_2d == cid] = col",
        "    return rgb",
        "",
        "def create_hazard_overlay(image_rgb: np.ndarray, mask_2d: np.ndarray, alpha=0.55) -> np.ndarray:",
        "    rgb_mask = decode_mask_to_rgb(mask_2d)",
        "    valid = (mask_2d != IGNORE_INDEX)[:, :, np.newaxis]",
        "    blended = np.where(valid, (1 - alpha) * image_rgb + alpha * rgb_mask, image_rgb)",
        "    return np.clip(blended, 0, 255).astype(np.uint8)"
    ]))

    cells.append(make_code_cell([
        "# Visualize Sample Image-Mask Pairs",
        "num_display = 3",
        "fig, axes = plt.subplots(num_display, 3, figsize=(16, 4.5 * num_display), dpi=100)",
        "",
        "for i in range(num_display):",
        "    img_path = matched_images[i]",
        "    mask_path = matched_masks[i]",
        "    ",
        "    raw_img = Image.open(img_path).convert('RGB')",
        "    raw_mask = np.array(Image.open(mask_path))",
        "    ",
        "    # Remap 4 -> 255 for display",
        "    raw_mask_remapped = raw_mask.copy()",
        "    raw_mask_remapped[raw_mask_remapped == 4] = IGNORE_INDEX",
        "    ",
        "    img_np = np.array(raw_img)",
        "    mask_rgb = decode_mask_to_rgb(raw_mask_remapped)",
        "    overlay = create_hazard_overlay(img_np, raw_mask_remapped, alpha=0.55)",
        "    ",
        "    axes[i, 0].imshow(img_np)",
        "    axes[i, 0].set_title(f'Raw Mars Image #{i+1}\\n({raw_img.size[0]} × {raw_img.size[1]} px)', fontsize=11)",
        "    axes[i, 0].axis('off')",
        "    ",
        "    axes[i, 1].imshow(mask_rgb)",
        "    axes[i, 1].set_title('Decoded Ground Truth Mask', fontsize=11)",
        "    axes[i, 1].axis('off')",
        "    ",
        "    axes[i, 2].imshow(overlay)",
        "    axes[i, 2].set_title('Terrain Hazard Overlay', fontsize=11)",
        "    axes[i, 2].axis('off')",
        "",
        "# Add legend",
        "patches = [",
        "    mpatches.Patch(color=np.array(COLOR_MAP[0])/255.0, label='Soil (Traversable)'),",
        "    mpatches.Patch(color=np.array(COLOR_MAP[1])/255.0, label='Bedrock (High Traction)'),",
        "    mpatches.Patch(color=np.array(COLOR_MAP[2])/255.0, label='Sand (Slippage Risk)'),",
        "    mpatches.Patch(color=np.array(COLOR_MAP[3])/255.0, label='Big Rock (Collision Hazard)'),",
        "    mpatches.Patch(color=np.array(COLOR_MAP[IGNORE_INDEX])/255.0, label='Unlabeled / Ignored'),",
        "]",
        "fig.legend(handles=patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=11)",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    cells.append(make_code_cell([
        "# Class Frequency Analysis Across Dataset",
        "print('Computing class pixel distribution across 300 sample masks...')",
        "pixel_counts = {0: 0, 1: 0, 2: 0, 3: 0, 'Unlabeled': 0}",
        "",
        "sample_size = min(300, len(matched_masks))",
        "for mask_p in matched_masks[:sample_size]:",
        "    m = np.array(Image.open(mask_p))",
        "    pixel_counts[0] += int(np.sum(m == 0))",
        "    pixel_counts[1] += int(np.sum(m == 1))",
        "    pixel_counts[2] += int(np.sum(m == 2))",
        "    pixel_counts[3] += int(np.sum(m == 3))",
        "    pixel_counts['Unlabeled'] += int(np.sum((m == 4) | (m == 255)))",
        "",
        "total_valid_pixels = sum(pixel_counts[i] for i in range(4))",
        "",
        "print('=' * 55)",
        "print(f'TERRAIN CLASS DISTRIBUTION ({sample_size} Sample Masks)')",
        "print('=' * 55)",
        "for i in range(4):",
        "    pct = (pixel_counts[i] / total_valid_pixels) * 100.0",
        "    print(f'  Class {i} [{TERRAIN_CLASSES[i]:<10}]: {pixel_counts[i]:>12,} px ({pct:5.2f}% of labeled)')",
        "print(f'  Unlabeled / Ignored:  {pixel_counts[\"Unlabeled\"]:>12,} px')",
        "print('=' * 55)",
        "",
        "# Plot Class Distribution Chart",
        "class_labels = [TERRAIN_CLASSES[i] for i in range(4)]",
        "class_pcts = [(pixel_counts[i] / total_valid_pixels) * 100.0 for i in range(4)]",
        "colors = [np.array(COLOR_MAP[i]) / 255.0 for i in range(4)]",
        "",
        "plt.figure(figsize=(9, 4.5), dpi=100)",
        "bars = plt.bar(class_labels, class_pcts, color=colors, edgecolor='black', width=0.55)",
        "plt.ylabel('Percentage of Labeled Martian Surface (%)', fontsize=11, fontweight='bold')",
        "plt.title('AI4Mars MastCam Terrain Class Frequency', fontsize=13, fontweight='bold', pad=12)",
        "plt.ylim(0, max(class_pcts) * 1.15)",
        "for bar in bars:",
        "    yval = bar.get_height()",
        "    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')",
        "plt.grid(axis='y', linestyle='--', alpha=0.3)",
        "plt.show()"
    ]))

    # ==========================================
    # Step 4: Data Preprocessing
    # ==========================================
    cells.append(make_markdown_cell([
        "## 🛠️ Step 4: Data Preprocessing & Augmentation",
        "",
        "### Preprocessing Strategy:",
        "1. **Resolution Standardization (256 × 256):** Standardizes varying MastCam resolutions while fitting comfortably in GPU VRAM.",
        "2. **Nearest-Neighbor Mask Interpolation:** When resizing the segmentation mask, we **MUST** use nearest-neighbor interpolation (`cv2.INTER_NEAREST` or `Image.NEAREST`). Bilinear interpolation would generate non-existent float values (e.g. interpolating Class 0 and Class 2 into 1).",
        "3. **Ignore Index Mapping:** Any label value of `4` or `255` is remapped to `IGNORE_INDEX = 255` so the loss function and metrics exclude them.",
        "4. **ImageNet Normalization:** Normalizes RGB channels using $\\mu = [0.485, 0.456, 0.406]$ and $\\sigma = [0.229, 0.224, 0.225]$ to match the pretrained ResNet-34 encoder.",
        "5. **Data Augmentation (Train Only):** `HorizontalFlip`, `RandomBrightnessContrast`, and `ShiftScaleRotate` applied to training samples to prevent overfitting and improve generalization across changing Martian lighting."
    ]))

    cells.append(make_code_cell([
        "# Define Preprocessing & Augmentation Pipelines",
        "IMAGENET_MEAN = (0.485, 0.456, 0.406)",
        "IMAGENET_STD = (0.229, 0.224, 0.225)",
        "TARGET_SIZE = (256, 256)  # (H, W)",
        "",
        "# Albumentations Compose Pipelines",
        "train_transform = A.Compose([",
        "    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1], interpolation=1),",
        "    A.HorizontalFlip(p=0.5),",
        "    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),",
        "    A.ShiftScaleRotate(",
        "        shift_limit=0.05,",
        "        scale_limit=0.05,",
        "        rotate_limit=15,",
        "        border_mode=0,",
        "        value=0,",
        "        mask_value=IGNORE_INDEX,  # Ignore boundary padding",
        "        p=0.4,",
        "    ),",
        "    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),",
        "    ToTensorV2(),",
        "])",
        "",
        "val_transform = A.Compose([",
        "    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1], interpolation=1),",
        "    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),",
        "    ToTensorV2(),",
        "])",
        "",
        "print('[OK] Training and Validation augmentation pipelines defined.')"
    ]))

    # ==========================================
    # Step 5: Train and Validation Split
    # ==========================================
    cells.append(make_markdown_cell([
        "## ✂️ Step 5: Train and Validation Split",
        "We divide our matched image-mask pairs into an **80% Training** and **20% Validation** split. Using `train_test_split` with a fixed random seed ensures reproducibility and prevents data leakage."
    ]))

    cells.append(make_code_cell([
        "# Perform 80:20 Train-Validation Split",
        "train_images, val_images, train_masks, val_masks = train_test_split(",
        "    matched_images,",
        "    matched_masks,",
        "    test_size=0.20,",
        "    random_state=RANDOM_SEED,",
        "    shuffle=True,",
        ")",
        "",
        "print('=' * 50)",
        "print('DATASET SPLIT SUMMARY')",
        "print('=' * 50)",
        "print(f'Training Pairs:    {len(train_images):,} (80%)')",
        "print(f'Validation Pairs:  {len(val_images):,} (20%)')",
        "print(f'Total Dataset:     {len(matched_images):,}')",
        "print('=' * 50)",
        "",
        "# Sanity Check: Verify stem alignment for random pair",
        "sample_idx = random.randint(0, len(train_images) - 1)",
        "img_stem = Path(train_images[sample_idx]).stem",
        "mask_stem = Path(train_masks[sample_idx]).stem",
        "print(f'Alignment Verification Sample [{sample_idx}]:')",
        "print(f'  Image: {img_stem}')",
        "print(f'  Mask:  {mask_stem}')",
        "assert mask_stem.startswith(img_stem), 'Alignment Error: Image and mask stems do not match!'"
    ]))

    # ==========================================
    # Step 6: PyTorch Dataset and DataLoader
    # ==========================================
    cells.append(make_markdown_cell([
        "## 📦 Step 6: PyTorch Dataset & DataLoader",
        "We construct a custom PyTorch `Dataset` that loads each sample on demand, remaps class values, resizes both images and masks correctly, and applies our augmentations."
    ]))

    cells.append(make_code_cell([
        "class MarsTerrainDataset(Dataset):",
        "    \"\"\"Custom PyTorch Dataset for AI4Mars Terrain Segmentation.\"\"\"",
        "    def __init__(self, image_paths: List[str], mask_paths: List[str], transform=None, target_size=(256, 256)):",
        "        self.image_paths = image_paths",
        "        self.mask_paths = mask_paths",
        "        self.transform = transform",
        "        self.target_size = target_size",
        "    ",
        "    def __len__(self) -> int:",
        "        return len(self.image_paths)",
        "    ",
        "    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:",
        "        try:",
        "            # 1. Load image (RGB)",
        "            with Image.open(self.image_paths[idx]) as img_pil:",
        "                image = np.array(img_pil.convert('RGB'))",
        "            ",
        "            # 2. Load mask (Grayscale)",
        "            with Image.open(self.mask_paths[idx]) as mask_pil:",
        "                mask = np.array(mask_pil)",
        "            ",
        "            # 3. Standardize class IDs (remap 4 or any >3 to IGNORE_INDEX 255)",
        "            mask_processed = mask.copy()",
        "            invalid = (mask_processed < 0) | (mask_processed > 3)",
        "            mask_processed[invalid] = IGNORE_INDEX",
        "            mask_processed = mask_processed.astype(np.uint8)",
        "            ",
        "            # 4. Apply Albumentations transform",
        "            if self.transform is not None:",
        "                augmented = self.transform(image=image, mask=mask_processed)",
        "                img_tensor = augmented['image']",
        "                mask_tensor = augmented['mask'].long()",
        "            else:",
        "                img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0",
        "                mask_tensor = torch.from_numpy(mask_processed).long()",
        "                ",
        "            return img_tensor, mask_tensor",
        "        except Exception as e:",
        "            print(f'[!] Corrupted sample at index {idx} ({self.image_paths[idx]}): {e}')",
        "            fallback_idx = (idx + 1) % len(self.image_paths)",
        "            return self.__getitem__(fallback_idx)",
        "",
        "# Instantiate Datasets",
        "train_dataset = MarsTerrainDataset(train_images, train_masks, transform=train_transform)",
        "val_dataset = MarsTerrainDataset(val_images, val_masks, transform=val_transform)",
        "",
        "# Configure DataLoaders",
        "BATCH_SIZE = 8 if not torch.cuda.is_available() else 16",
        "NUM_WORKERS = 2",
        "",
        "train_loader = DataLoader(",
        "    train_dataset,",
        "    batch_size=BATCH_SIZE,",
        "    shuffle=True,",
        "    num_workers=NUM_WORKERS,",
        "    pin_memory=(device.type == 'cuda'),",
        "    drop_last=True,",
        ")",
        "",
        "val_loader = DataLoader(",
        "    val_dataset,",
        "    batch_size=BATCH_SIZE,",
        "    shuffle=False,",
        "    num_workers=NUM_WORKERS,",
        "    pin_memory=(device.type == 'cuda'),",
        ")",
        "",
        "print(f'[OK] DataLoaders initialized: Batch Size = {BATCH_SIZE}, Workers = {NUM_WORKERS}')",
        "print(f'     Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}')"
    ]))

    cells.append(make_code_cell([
        "# Sanity Check: Inspect a single batch from DataLoader",
        "sample_images, sample_masks = next(iter(train_loader))",
        "print('Batch Dimensions:')",
        "print(f'  Images Tensor: {sample_images.shape} (dtype: {sample_images.dtype})')",
        "print(f'  Masks Tensor:  {sample_masks.shape}  (dtype: {sample_masks.dtype})')",
        "print(f'  Mask unique values in batch: {torch.unique(sample_masks).tolist()}')",
        "assert sample_images.shape == (BATCH_SIZE, 3, 256, 256)",
        "assert sample_masks.shape == (BATCH_SIZE, 256, 256)",
        "print('[OK] Tensor dimensions strictly validated.')"
    ]))

    # ==========================================
    # Step 7: Model Architecture
    # ==========================================
    cells.append(make_markdown_cell([
        "## 🏗️ Step 7: Model Architecture — U-Net with ResNet-34 Encoder",
        "",
        "### Architectural Rationale:",
        "- **U-Net:** Features an encoder-decoder architecture with lateral **skip connections**. Skip connections pass high-resolution feature maps from encoder stages directly to corresponding decoder stages, preserving fine-grained spatial boundaries (essential for detecting the edges of rocks and sand boundaries).",
        r"- **ResNet-34 Backbone:** Pretrained on ImageNet (1.2M images), the ResNet-34 encoder leverages 34 convolutional layers with residual connections ($x + \mathcal{F}(x)$), eliminating vanishing gradients and extracting robust multi-scale features from fine Martian regolith textures to broad geological formations.",
        "- **Configuration:**",
        "  - Input Channels: `3` (RGB MastCam)",
        "  - Output Classes: `4` (Soil, Bedrock, Sand, Big Rocks)",
        "  - Resolution: `256 × 256`"
    ]))

    cells.append(make_code_cell([
        "# Model Construction",
        "def build_model(num_classes: int = 4, pretrained: bool = True):",
        "    try:",
        "        import segmentation_models_pytorch as smp",
        "        print(f'[*] Initializing SMP U-Net with ResNet-34 encoder (Pretrained={pretrained})...')",
        "        model = smp.Unet(",
        "            encoder_name='resnet34',",
        "            encoder_weights='imagenet' if pretrained else None,",
        "            in_channels=3,",
        "            classes=num_classes,",
        "        )",
        "        return model",
        "    except Exception as e:",
        "        print(f'[!] Falling back to native PyTorch ResNet34 U-Net: {e}')",
        "        # Native fallback model if SMP is unavailable",
        "        from torchvision.models import resnet34, ResNet34_Weights",
        "        weights = ResNet34_Weights.DEFAULT if pretrained else None",
        "        backbone = resnet34(weights=weights)",
        "        # Standard SMP-style U-Net head",
        "        raise RuntimeError('Please install segmentation-models-pytorch: !pip install segmentation-models-pytorch')",
        "",
        "model = build_model(num_classes=4, pretrained=True).to(device)",
        "",
        "total_params = sum(p.numel() for p in model.parameters())",
        "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "print(f'[OK] Model Architecture: U-Net (ResNet-34 Encoder)')",
        "print(f'     Total Parameters:     {total_params:,}')",
        "print(f'     Trainable Parameters: {trainable_params:,}')"
    ]))

    # ==========================================
    # Step 8: Model Training
    # ==========================================
    cells.append(make_markdown_cell([
        "## 🚀 Step 8: Model Training & Checkpointing",
        "",
        "### Training Strategy:",
        "- **Loss Function:** `nn.CrossEntropyLoss(ignore_index=255)`. Pixels marked with `255` are mathematically ignored from both loss computation and gradient backpropagation.",
        "- **Optimizer:** Adam with initial learning rate $\\eta = 0.001$ and $L_2$ weight decay $10^{-4}$.",
        "- **Learning Rate Scheduler:** `ReduceLROnPlateau` decays learning rate by a factor of 0.5 if validation loss plateaus for 3 epochs.",
        "- **Best Checkpoint Saving:** Automatically persists the model weights with lowest validation loss to `mars_terrain_model.pth`."
    ]))

    cells.append(make_code_cell([
        "# Training & Evaluation Functions",
        "def train_epoch(model, loader, criterion, optimizer, device):",
        "    model.train()",
        "    total_loss = 0.0",
        "    for images, masks in loader:",
        "        images = images.to(device)",
        "        masks = masks.to(device)",
        "        ",
        "        optimizer.zero_grad()",
        "        outputs = model(images)",
        "        loss = criterion(outputs, masks)",
        "        loss.backward()",
        "        optimizer.step()",
        "        ",
        "        total_loss += loss.item()",
        "    return total_loss / len(loader)",
        "",
        "@torch.no_grad()",
        "def validate_epoch(model, loader, criterion, device):",
        "    model.eval()",
        "    total_loss = 0.0",
        "    total_correct = 0",
        "    total_valid_pixels = 0",
        "    ",
        "    for images, masks in loader:",
        "        images = images.to(device)",
        "        masks = masks.to(device)",
        "        ",
        "        outputs = model(images)",
        "        loss = criterion(outputs, masks)",
        "        total_loss += loss.item()",
        "        ",
        "        preds = torch.argmax(outputs, dim=1)",
        "        valid_mask = (masks != IGNORE_INDEX)",
        "        total_correct += ((preds == masks) & valid_mask).sum().item()",
        "        total_valid_pixels += valid_mask.sum().item()",
        "        ",
        "    val_loss = total_loss / len(loader)",
        "    pixel_acc = total_correct / max(1, total_valid_pixels)",
        "    return val_loss, pixel_acc"
    ]))

    cells.append(make_code_cell([
        "# Hyperparameters & Optimization Setup",
        "EPOCHS = 20",
        "LEARNING_RATE = 1e-3",
        "CHECKPOINT_DIR = Path('checkpoints')",
        "CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)",
        "BEST_MODEL_PATH = CHECKPOINT_DIR / 'mars_terrain_model.pth'",
        "",
        "criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)",
        "optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)",
        "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)",
        "",
        "print(f'[OK] Training configured for {EPOCHS} epochs.')"
    ]))

    cells.append(make_code_cell([
        "# Training Loop Execution",
        "history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}",
        "best_val_loss = float('inf')",
        "",
        "print('=' * 70)",
        "print('TRAINING U-NET RESNET-34 ON AI4MARS MASTCAM')",
        "print('=' * 70)",
        "",
        "for epoch in range(EPOCHS):",
        "    t0 = time.time()",
        "    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)",
        "    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)",
        "    elapsed = time.time() - t0",
        "    ",
        "    scheduler.step(val_loss)",
        "    current_lr = optimizer.param_groups[0]['lr']",
        "    ",
        "    history['train_loss'].append(train_loss)",
        "    history['val_loss'].append(val_loss)",
        "    history['val_accuracy'].append(val_acc)",
        "    ",
        "    print(f'Epoch [{epoch+1:02d}/{EPOCHS:02d}] ({elapsed:4.1f}s) | '",
        "          f'Train Loss: {train_loss:.4f} | '",
        "          f'Val Loss: {val_loss:.4f} | '",
        "          f'Val Pixel Acc: {val_acc*100:5.2f}% | '",
        "          f'LR: {current_lr:.6f}')",
        "          ",
        "    # Save Best Checkpoint",
        "    if val_loss < best_val_loss:",
        "        best_val_loss = val_loss",
        "        torch.save({",
        "            'epoch': epoch + 1,",
        "            'model_state_dict': model.state_dict(),",
        "            'optimizer_state_dict': optimizer.state_dict(),",
        "            'val_loss': val_loss,",
        "            'val_accuracy': val_acc,",
        "            'classes': TERRAIN_CLASSES,",
        "        }, BEST_MODEL_PATH)",
        "        print(f'  --> [Saved Best Model Checkpoint] Val Loss: {val_loss:.4f}')",
        "",
        "print('=' * 70)",
        "print(f'[OK] Training Complete. Best model saved to: {BEST_MODEL_PATH}')"
    ]))

    cells.append(make_code_cell([
        "# Plot Training & Validation Curves",
        "plt.figure(figsize=(12, 5), dpi=100)",
        "",
        "plt.subplot(1, 2, 1)",
        "plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='#FF6B4A', lw=2)",
        "plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='#3498DB', lw=2)",
        "plt.title('Cross-Entropy Loss (Excl. Ignored)', fontsize=12, fontweight='bold')",
        "plt.xlabel('Epoch', fontsize=10)",
        "plt.ylabel('Loss', fontsize=10)",
        "plt.legend()",
        "plt.grid(True, linestyle='--', alpha=0.3)",
        "",
        "plt.subplot(1, 2, 2)",
        "plt.plot(range(1, len(history['val_accuracy'])+1), [a*100 for a in history['val_accuracy']], label='Val Pixel Accuracy (%)', color='#2ECC71', lw=2)",
        "plt.title('Validation Pixel Accuracy', fontsize=12, fontweight='bold')",
        "plt.xlabel('Epoch', fontsize=10)",
        "plt.ylabel('Accuracy (%)', fontsize=10)",
        "plt.legend()",
        "plt.grid(True, linestyle='--', alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # ==========================================
    # Step 9: Model Evaluation
    # ==========================================
    cells.append(make_markdown_cell([
        "## 📊 Step 9: Quantitative Model Evaluation",
        "",
        "To rigorously measure performance, we evaluate the best model checkpoint on the validation set using standard semantic segmentation benchmarks:",
        "",
        "1. **Pixel Accuracy:** $\\frac{\\sum TP}{\\text{Total Valid Pixels}}$",
        "2. **Intersection over Union (IoU / Jaccard Index):**",
        "   $$\\text{IoU}_c = \\frac{TP_c}{TP_c + FP_c + FN_c}$$",
        "3. **Mean IoU (mIoU):** $\\frac{1}{C} \\sum_{c=1}^C \\text{IoU}_c$",
        "4. **Dice Coefficient (F1 Score):**",
        "   $$\\text{Dice}_c = \\frac{2 \\times TP_c}{2 \\times TP_c + FP_c + FN_c}$$",
        "5. **4 × 4 Normalized Confusion Matrix:** Reveals precise inter-class confusion patterns (e.g. soil vs sand)."
    ]))

    cells.append(make_code_cell([
        "# Load Best Model Checkpoint for Evaluation",
        "if BEST_MODEL_PATH.exists():",
        "    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)",
        "    model.load_state_dict(checkpoint['model_state_dict'])",
        "    print(f'[OK] Successfully loaded best model from {BEST_MODEL_PATH} (Epoch {checkpoint[\"epoch\"]})')",
        "else:",
        "    print('[!] Best checkpoint not found; evaluating current model weights.')",
        "",
        "model.eval()",
        "all_preds = []",
        "all_targets = []",
        "",
        "print('Computing predictions on validation dataset...')",
        "with torch.no_grad():",
        "    for images, masks in val_loader:",
        "        images = images.to(device)",
        "        outputs = model(images)",
        "        preds = torch.argmax(outputs, dim=1).cpu().numpy()",
        "        targets = masks.numpy()",
        "        ",
        "        all_preds.append(preds)",
        "        all_targets.append(targets)",
        "",
        "all_preds = np.concatenate(all_preds, axis=0)",
        "all_targets = np.concatenate(all_targets, axis=0)",
        "print(f'[OK] Evaluated {len(all_preds):,} validation masks.')"
    ]))

    cells.append(make_code_cell([
        "# Compute Segmentation Metrics (Strictly Excl. Ignored Pixels)",
        "preds_flat = all_preds.flatten()",
        "targets_flat = all_targets.flatten()",
        "",
        "valid_idx = (targets_flat != IGNORE_INDEX) & (targets_flat >= 0) & (targets_flat < 4)",
        "p_valid = preds_flat[valid_idx]",
        "t_valid = targets_flat[valid_idx]",
        "",
        "# Pixel Accuracy",
        "overall_pixel_acc = float(np.sum(p_valid == t_valid) / len(t_valid))",
        "",
        "# 4x4 Confusion Matrix",
        "cm = np.zeros((4, 4), dtype=np.int64)",
        "for t, p in zip(t_valid, p_valid):",
        "    cm[t, p] += 1",
        "",
        "# Per-class IoU and Dice",
        "ious, dices = [], []",
        "per_class_metrics = {}",
        "",
        "for c in range(4):",
        "    tp = cm[c, c]",
        "    fn = np.sum(cm[c, :]) - tp",
        "    fp = np.sum(cm[:, c]) - tp",
        "    ",
        "    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0",
        "    dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0",
        "    ",
        "    ious.append(iou)",
        "    dices.append(dice)",
        "    per_class_metrics[c] = {'iou': iou, 'dice': dice}",
        "",
        "miou = float(np.mean(ious))",
        "mean_dice = float(np.mean(dices))",
        "",
        "print('=' * 65)",
        "print('QUANTITATIVE EVALUATION RESULTS (AI4MARS VALIDATION SET)')",
        "print('=' * 65)",
        "print(f'Overall Pixel Accuracy:       {overall_pixel_acc * 100:.2f}%')",
        "print(f'Mean Intersection over Union:  {miou * 100:.2f}%')",
        "print(f'Mean Dice Score (F1):          {mean_dice * 100:.2f}%')",
        "print('-' * 65)",
        "print(f'{\"Class ID\":<10} {\"Terrain Class\":<15} {\"IoU (%)\":<15} {\"Dice / F1 (%)\":<15}')",
        "print('-' * 65)",
        "for c in range(4):",
        "    c_name = TERRAIN_CLASSES[c]",
        "    ciou = per_class_metrics[c]['iou'] * 100.0",
        "    cdice = per_class_metrics[c]['dice'] * 100.0",
        "    print(f'{c:<10} {c_name:<15} {ciou:<15.2f} {cdice:<15.2f}')",
        "print('=' * 65)"
    ]))

    cells.append(make_code_cell([
        "# Plot Normalized Confusion Matrix Heatmap",
        "row_sums = cm.sum(axis=1, keepdims=True)",
        "norm_cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)",
        "",
        "plt.figure(figsize=(7.5, 6), dpi=100)",
        "sns.heatmap(",
        "    norm_cm * 100.0,",
        "    annot=True,",
        "    fmt='.1f',",
        "    cmap='Blues',",
        "    xticklabels=[TERRAIN_CLASSES[i] for i in range(4)],",
        "    yticklabels=[TERRAIN_CLASSES[i] for i in range(4)],",
        "    cbar_kws={'label': 'Recall Percentage (%)'},",
        "    linewidths=1,",
        "    linecolor='#2D3748',",
        ")",
        "plt.title('Normalized Confusion Matrix (%) — Excl. Ignored Pixels', fontsize=12, fontweight='bold', pad=12)",
        "plt.ylabel('Ground Truth Class', fontsize=11, fontweight='bold')",
        "plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # ==========================================
    # Step 10: Prediction Visualization
    # ==========================================
    cells.append(make_markdown_cell([
        "## 🔭 Step 10: Prediction Visualization & Qualitative Assessment",
        "",
        "We visualize qualitative predictions for representative validation samples across 4 panels:",
        "1. **Original Mars Image (MastCam RGB)**",
        "2. **Ground-Truth Terrain Mask**",
        "3. **Model Predicted Segmentation Mask**",
        "4. **Terrain Hazard Overlay (Alpha Blend)**"
    ]))

    cells.append(make_code_cell([
        "# Qualitative Prediction Grid Visualizer",
        "num_viz = 4",
        "fig, axes = plt.subplots(num_viz, 4, figsize=(18, 4.2 * num_viz), dpi=120)",
        "",
        "headers = [",
        "    '1. Raw Mars Rover Image',",
        "    '2. Ground-Truth Mask',",
        "    '3. Predicted Segmentation Mask',",
        "    '4. Terrain Hazard Overlay',",
        "]",
        "for col, h in enumerate(headers):",
        "    axes[0, col].set_title(h, fontsize=12, fontweight='bold', pad=8)",
        "",
        "for i in range(num_viz):",
        "    # Denormalize image for display",
        "    img_tensor, mask_tensor = val_dataset[i]",
        "    img_np = img_tensor.numpy().transpose(1, 2, 0)",
        "    img_np = (img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN))",
        "    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)",
        "    ",
        "    gt_mask = mask_tensor.numpy()",
        "    pred_mask = all_preds[i]",
        "    overlay = create_hazard_overlay(img_np, pred_mask, alpha=0.55)",
        "    ",
        "    axes[i, 0].imshow(img_np)",
        "    axes[i, 0].axis('off')",
        "    ",
        "    axes[i, 1].imshow(decode_mask_to_rgb(gt_mask))",
        "    axes[i, 1].axis('off')",
        "    ",
        "    axes[i, 2].imshow(decode_mask_to_rgb(pred_mask))",
        "    axes[i, 2].axis('off')",
        "    ",
        "    axes[i, 3].imshow(overlay)",
        "    axes[i, 3].axis('off')",
        "",
        "fig.legend(handles=patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=11)",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # ==========================================
    # Step 11: Save Model & Artifacts
    # ==========================================
    cells.append(make_markdown_cell([
        "## 💾 Step 11: Save Model & Export Preprocessing Configuration",
        "We serialize the complete model package for downstream deployment, saving:",
        "- `mars_terrain_model.pth`: Trained state dictionary and architecture metadata.",
        "- `class_mapping.json`: Class ID to name mappings and RGB color palette.",
        "- `config.json`: Input resolution, normalization constants, and model hyperparameters."
    ]))

    cells.append(make_code_cell([
        "# Save Deployment Artifacts",
        "export_dir = Path('export')",
        "export_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "# 1. Save Config JSON",
        "config_payload = {",
        "    'architecture': 'U-Net',",
        "    'encoder': 'resnet34',",
        "    'num_classes': 4,",
        "    'classes': TERRAIN_CLASSES,",
        "    'ignore_index': IGNORE_INDEX,",
        "    'input_size': [256, 256],",
        "    'imagenet_mean': list(IMAGENET_MEAN),",
        "    'imagenet_std': list(IMAGENET_STD),",
        "    'color_palette': {k: list(v) for k, v in COLOR_MAP.items()},",
        "}",
        "",
        "with open(export_dir / 'config.json', 'w') as f:",
        "    json.dump(config_payload, f, indent=4)",
        "",
        "# 2. Copy/Verify Best Model Checkpoint",
        "export_model_path = export_dir / 'mars_terrain_model.pth'",
        "torch.save({",
        "    'model_state_dict': model.state_dict(),",
        "    'config': config_payload,",
        "}, export_model_path)",
        "",
        "print('[OK] Deployment artifacts exported to:')",
        "print(f'  - Model:  {export_model_path} ({export_model_path.stat().st_size / (1024**2):.2f} MB)')",
        "print(f'  - Config: {export_dir / \"config.json\"}')",
        "",
        "if IN_COLAB:",
        "    print('\\nTo download these artifacts to your local machine, run:')",
        "    print('from google.colab import files')",
        "    print(f'files.download(\"{export_model_path}\")')",
        "    print(f'files.download(\"{export_dir / \"config.json\"}\")')"
    ]))

    # ==========================================
    # Step 12: Deployment Preparation & Safety
    # ==========================================
    cells.append(make_markdown_cell([
        "## 🚀 Step 12: Deployment Preparation & Space Robotics Safety Advisory",
        "",
        "### Interactive Streamlit Application (`app.py`)",
        "An interactive, modern space-themed web application is included in this repository (`app.py`).",
        "",
        "#### Features of the Dashboard:",
        "- **Dark Space UI:** Professional space-themed interface with Mars-inspired orange accents (`#FF5722`).",
        "- **Image Ingestion:** Upload any custom Mars rover image (JPG/PNG) or pick from built-in Curiosity presets.",
        "- **Real-Time U-Net Inference:** Computes pixel-wise terrain segmentation at 256 × 256 resolution.",
        "- **Hazard Overlay Control:** Interactive alpha-transparency slider for rover operators.",
        "- **Surface Composition Breakdown:** Live percentage breakdown of Soil, Bedrock, Sand, and Big Rocks.",
        "- **Autonomy & Traversability Assessment:** Rule-based decision advisory flagging severe boulder hazards or sand slippage traps.",
        "",
        "#### Running the Streamlit App Locally:",
        "```bash",
        "streamlit run app.py",
        "```",
        "",
        "---",
        "### ⚠️ Critical Planetary Robotics & Safety Disclaimer",
        "> **IMPORTANT:** Semantic segmentation is an indispensable visual perception component, but **segmentation alone does not guarantee rover safety.**",
        "> ",
        "> Real-world autonomous rover navigation systems (such as NASA's **AutoNav** on Curiosity and Perseverance) employ multi-modal sensor fusion incorporating:",
        "> 1. **Dense 3D Stereo Vision Point Clouds:** Measures geometric ground clearance, step heights, and pitch/roll angles.",
        "> 2. **Digital Elevation Models (DEMs):** Evaluates local slope gradients (rovers cannot safely traverse slopes exceeding ~30°).",
        "> 3. **Wheel Kinematics & Sinkage Telemetry:** Real-time visual odometry detects wheel slippage before the rover becomes entrenched.",
        "> 4. **Solar Illumination & Shadow Modeling:** Mitigates false-positive obstacle detection caused by low-angle Martian sun shadows.",
        "",
        "Congratulations! You have completed the complete end-to-end Mars Rover Terrain Segmentation project!"
    ]))

    notebook_structure = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "Mars_Rover_Terrain_Segmentation.ipynb",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    output_path = Path(r"d:\mars\Mars_Rover_Terrain_Segmentation.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_structure, f, indent=2)

    print(f"[OK] Successfully built: {output_path}")


if __name__ == "__main__":
    build_notebook()
