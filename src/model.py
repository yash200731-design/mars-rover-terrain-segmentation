"""
Model Architecture Definition for Mars Rover Terrain Segmentation
U-Net with ResNet-34 Encoder Backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block:
    Upsamples feature map, concatenates skip connection from encoder,
    and applies two 3x3 convolutions with BatchNorm and ReLU.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Bilinear upsampling to match skip feature map size
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class NativeResNet34UNet(nn.Module):
    """
    Pure PyTorch implementation of U-Net with ResNet-34 encoder.
    Provides offline/zero-dependency capability if segmentation-models-pytorch is absent.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            resnet = resnet34(weights=weights)
        except Exception:
            from torchvision.models import resnet34
            resnet = resnet34(weights=None)

        # Encoder stages
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # /4

        self.layer1 = resnet.layer1  # 64 channels, /4
        self.layer2 = resnet.layer2  # 128 channels, /8
        self.layer3 = resnet.layer3  # 256 channels, /16
        self.layer4 = resnet.layer4  # 512 channels, /32

        # Decoder stages
        # block4: takes layer4 (512) and layer3 skip (256) -> 256
        self.dec4 = DecoderBlock(512, 256, 256)
        # block3: takes dec4 (256) and layer2 skip (128) -> 128
        self.dec3 = DecoderBlock(256, 128, 128)
        # block2: takes dec3 (128) and layer1 skip (64) -> 64
        self.dec2 = DecoderBlock(128, 64, 64)
        # block1: takes dec2 (64) and conv1 skip (64) -> 32
        self.dec1 = DecoderBlock(64, 64, 32)
        # final upsample to original resolution: /2 to /1 -> 16
        self.dec0 = DecoderBlock(32, 0, 16)

        # Final classification head: 16 -> num_classes (4)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder forward pass
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)                   # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)                    # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)                    # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)                    # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                    # [B, 512, H/32, W/32]

        # Decoder forward pass with skip connections
        d4 = self.dec4(x4, x3)                  # [B, 256, H/16, W/16]
        d3 = self.dec3(d4, x2)                  # [B, 128, H/8, W/8]
        d2 = self.dec2(d3, x1)                  # [B, 64, H/4, W/4]
        d1 = self.dec1(d2, x0)                  # [B, 32, H/2, W/2]
        d0 = self.dec0(d1, None)                # [B, 16, H, W]

        logits = self.final_conv(d0)            # [B, num_classes, H, W]
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)
        return logits


def build_unet_resnet34(
    num_classes: int = 4,
    pretrained: bool = True,
    in_channels: int = 3,
    force_native: bool = False
) -> nn.Module:
    """
    Factory function to build a U-Net with ResNet-34 encoder.
    
    Prefers segmentation_models_pytorch if installed, otherwise seamlessly falls back
    to NativeResNet34UNet.
    """
    if HAS_SMP and not force_native:
        print(f"[Model Builder] Initializing SMP U-Net with ResNet-34 encoder (Pretrained={pretrained}, Classes={num_classes})")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        print(f"[Model Builder] Initializing Native PyTorch U-Net with ResNet-34 backbone (Classes={num_classes})")
        model = NativeResNet34UNet(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
        )
    return model
