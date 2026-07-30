"""
GeoSenseAI Change Detector

Purpose:
    Fuse multi-scale feature maps from two satellite images
    and prepare them for decoding into a change mask.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeatureFusionBlock(nn.Module):
    """
    Fuse Before and After Features
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = ConvBlock(channels * 2, channels)

    def forward(self, before, after):
        x = torch.cat([before, after], dim=1)
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsample + Skip Connection + Conv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


# ============================================================
# Change Detector
# ============================================================

class ChangeDetector(nn.Module):
    """
    Complete Change Detection Network
    """

    def __init__(self):
        super().__init__()

        # Feature Fusion
        self.fuse1 = FeatureFusionBlock(256)
        self.fuse2 = FeatureFusionBlock(512)
        self.fuse3 = FeatureFusionBlock(1024)
        self.fuse4 = FeatureFusionBlock(2048)

        # Decoder
        self.up3 = UpBlock(
            in_channels=2048,
            skip_channels=1024,
            out_channels=1024
        )

        self.up2 = UpBlock(
            in_channels=1024,
            skip_channels=512,
            out_channels=512
        )

        self.up1 = UpBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        # Final refinement
        self.final_conv = ConvBlock(
            256,
            64
        )

        # Segmentation Head
        self.seg_head = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(
        self,
        before_features,
        after_features
    ):

        # Feature Fusion
        f1 = self.fuse1(before_features[0], after_features[0])
        f2 = self.fuse2(before_features[1], after_features[1])
        f3 = self.fuse3(before_features[2], after_features[2])
        f4 = self.fuse4(before_features[3], after_features[3])

        # Decoder
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        x = self.final_conv(d1)

        mask = self.seg_head(x)

        # Upsample to original image size
        mask = F.interpolate(
            mask,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        return mask