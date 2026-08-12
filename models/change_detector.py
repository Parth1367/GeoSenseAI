"""
GeoSenseAI Change Detector V2

Purpose:
    Fuse multi-scale features from two satellite images
    using temporal difference and channel attention.

Backbone:
    ResNet-18

Feature channels:
    64
    128
    256
    512

V2 Improvements:
    1. Explicit absolute feature difference
    2. Before + After + Difference fusion
    3. Channel attention
    4. Multi-scale decoder
"""


from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Conv Block
# ============================================================

class ConvBlock(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Channel Attention
# ============================================================

class ChannelAttention(nn.Module):
    """
    Channel attention module.

    Learns which feature channels are important
    for detecting changes.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8
    ):

        super().__init__()

        hidden_channels = max(
            channels // reduction,
            8
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.attention = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=True
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=True
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.pool(x)

        weights = self.attention(
            weights
        )

        return x * weights


# ============================================================
# V2 Feature Fusion
# ============================================================

class FeatureFusionBlockV2(nn.Module):
    """
    Fuse Before and After features using:

        Before
        After
        |Before - After|

    Then apply channel attention.
    """

    def __init__(
        self,
        channels: int
    ):

        super().__init__()

        # Before + After + Difference
        input_channels = channels * 3

        self.conv = ConvBlock(
            input_channels,
            channels
        )

        self.attention = ChannelAttention(
            channels
        )

    def forward(
        self,
        before,
        after
    ):

        # ----------------------------------------------------
        # Explicit temporal difference
        # ----------------------------------------------------

        difference = torch.abs(
            before - after
        )

        # ----------------------------------------------------
        # Concatenate temporal information
        # ----------------------------------------------------

        x = torch.cat(
            [
                before,
                after,
                difference
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Feature refinement
        # ----------------------------------------------------

        x = self.conv(x)

        # ----------------------------------------------------
        # Channel attention
        # ----------------------------------------------------

        x = self.attention(x)

        return x


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(nn.Module):
    """
    Upsample decoder feature and fuse with
    corresponding encoder skip feature.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        # ----------------------------------------------------
        # Upsample
        # ----------------------------------------------------

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Skip connection
        # ----------------------------------------------------

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Refinement
        # ----------------------------------------------------

        x = self.conv(x)

        return x


# ============================================================
# Change Detector V2
# ============================================================

class ChangeDetector(nn.Module):
    """
    GeoSenseAI V2 Change Detection Module.

    Input:

        Before features:
            f1 -> [B, 64, 64, 64]
            f2 -> [B, 128, 32, 32]
            f3 -> [B, 256, 16, 16]
            f4 -> [B, 512, 8, 8]

        After features:
            same dimensions

    Output:

        Change mask:
            [B, 1, 256, 256]
    """

    def __init__(self):

        super().__init__()

        # ====================================================
        # Multi-scale temporal fusion
        # ====================================================

        self.fuse1 = FeatureFusionBlockV2(
            64
        )

        self.fuse2 = FeatureFusionBlockV2(
            128
        )

        self.fuse3 = FeatureFusionBlockV2(
            256
        )

        self.fuse4 = FeatureFusionBlockV2(
            512
        )

        # ====================================================
        # Decoder
        # ====================================================

        self.decoder3 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.decoder2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.decoder1 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        # ====================================================
        # Final prediction head
        # ====================================================

        self.final_conv = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        before_features: List[torch.Tensor],
        after_features: List[torch.Tensor]
    ):

        # ====================================================
        # V2 Temporal Feature Fusion
        # ====================================================

        f1 = self.fuse1(
            before_features[0],
            after_features[0]
        )

        f2 = self.fuse2(
            before_features[1],
            after_features[1]
        )

        f3 = self.fuse3(
            before_features[2],
            after_features[2]
        )

        f4 = self.fuse4(
            before_features[3],
            after_features[3]
        )

        # ====================================================
        # Decoder
        # ====================================================

        x = self.decoder3(
            f4,
            f3
        )

        x = self.decoder2(
            x,
            f2
        )

        x = self.decoder1(
            x,
            f1
        )

        # ====================================================
        # 64x64 -> 256x256
        # ====================================================

        x = F.interpolate(
            x,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # ====================================================
        # Final change mask
        # ====================================================

        mask = self.final_conv(
            x
        )

        return mask