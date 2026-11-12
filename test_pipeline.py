import sys
from pathlib import Path
import torch

from src.dataset import match_image_mask_pairs, MarsTerrainDataset, get_transforms
from src.model import build_unet_resnet34
from src.utils import TERRAIN_CLASSES, IGNORE_INDEX, calculate_metrics

print('1. Testing Pair Matcher...')
img_dir = Path(r'D:\mars\ai4mars-dataset-merged-0.6\msl\mcam\images')
lbl_dir = Path(r'D:\mars\ai4mars-dataset-merged-0.6\msl\mcam\labels\train')
matched_imgs, matched_lbls, unmatched_imgs, unmatched_lbls = match_image_mask_pairs(img_dir, lbl_dir, verbose=False)
print(f'   Matched pairs: {len(matched_imgs)}')
assert len(matched_imgs) == 9099, f'Expected 9099 pairs, got {len(matched_imgs)}'

print('2. Testing Dataset with single sample...')
dataset = MarsTerrainDataset(matched_imgs[:5], matched_lbls[:5], target_size=(256, 256))
img, mask = dataset[0]
print(f'   Image tensor shape: {img.shape}, dtype: {img.dtype}')
print(f'   Mask tensor shape: {mask.shape}, dtype: {mask.dtype}')
print(f'   Unique mask values: {torch.unique(mask).tolist()}')
assert img.shape == (3, 256, 256)
assert mask.shape == (256, 256)

print('3. Testing Model Architecture (U-Net with ResNet-34)...')
model = build_unet_resnet34(num_classes=4, pretrained=False, force_native=True)
dummy_input = torch.randn(2, 3, 256, 256)
output = model(dummy_input)
print(f'   Model output shape: {output.shape}')
assert output.shape == (2, 4, 256, 256)

print('4. Testing Metric Calculation...')
dummy_preds = torch.argmax(output, dim=1).numpy()
dummy_targets = mask.unsqueeze(0).repeat(2, 1, 1).numpy()
metrics = calculate_metrics(dummy_preds, dummy_targets, num_classes=4, ignore_index=IGNORE_INDEX)
print(f"   Calculated pixel accuracy: {metrics['pixel_accuracy']:.4f}")
print(f"   Calculated mIoU: {metrics['miou']:.4f}")

print('[ALL PIPELINE TESTS PASSED PERFECTLY!]')
