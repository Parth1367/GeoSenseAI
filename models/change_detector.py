"""
GeoSenseAI V5 Change Detector

V5 Improvements:
    1. Explicit temporal difference
    2. Temporal interaction features
    3. Channel attention
    4. Spatial attention
    5. Multi-scale decoder
    6. Transformer bottleneck refinement (BiT-style)
       — applied only at the deepest feature level (8x8 = 64
       tokens), where global self-attention is cheap enough
       for a 4GB GPU but still lets the model reason about
       long-range spatial relationships that pure CNNs miss.

Backbone:
    ResNet-18

Feature channels:
    64
    128
    256
    512
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
    Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU

    Uses GroupNorm instead of BatchNorm because training
    runs at batch_size=1 (GTX 1650 4GB constraint), where
    BatchNorm statistics are unreliable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                min(groups, out_channels),
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
                bias=False,
            ),

            nn.GroupNorm(
                min(groups, out_channels),
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Channel Attention
# ============================================================

class ChannelAttention(nn.Module):
    """
    Channel attention:
    learns which feature channels are important.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.attention = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=True,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        weights = self.pool(x)

        weights = self.attention(
            weights
        )

        return x * weights


# ============================================================
# Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):
    """
    Spatial attention:
    learns WHERE the important change regions are.
    """

    def __init__(
        self,
        kernel_size: int = 7,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_map = torch.max(
            x,
            dim=1,
            keepdim=True,
        ).values

        attention = torch.cat(
            [
                avg_map,
                max_map,
            ],
            dim=1,
        )

        attention = self.conv(
            attention
        )

        attention = self.sigmoid(
            attention
        )

        return x * attention


# ============================================================
# V5 Temporal Fusion
# ============================================================

class TemporalFusionBlockV5(nn.Module):
    """
    V5 temporal fusion.

    Instead of only:
        Before + After + |Before - After|

    we additionally use:
        Before * After

    The product provides an explicit feature interaction
    between the two temporal observations.

    Fusion:
        Before
        After
        Absolute Difference
        Product Interaction
            ↓
        Conv refinement
            ↓
        Channel Attention
            ↓
        Spatial Attention
    """

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        input_channels = channels * 4

        self.fusion = ConvBlock(
            input_channels,
            channels,
        )

        self.channel_attention = ChannelAttention(
            channels
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=7
        )

        # Residual projection
        self.residual = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            min(8, channels),
            channels,
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        before,
        after,
    ):

        # ----------------------------------------------------
        # Temporal difference
        # ----------------------------------------------------

        difference = torch.abs(
            before - after
        )

        # ----------------------------------------------------
        # Temporal interaction
        # ----------------------------------------------------

        interaction = (
            before * after
        )

        # ----------------------------------------------------
        # Combine temporal information
        # ----------------------------------------------------

        x = torch.cat(
            [
                before,
                after,
                difference,
                interaction,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Feature refinement
        # ----------------------------------------------------

        x = self.fusion(
            x
        )

        # ----------------------------------------------------
        # Channel attention
        # ----------------------------------------------------

        x = self.channel_attention(
            x
        )

        # ----------------------------------------------------
        # Spatial attention
        # ----------------------------------------------------

        x = self.spatial_attention(
            x
        )

        # ----------------------------------------------------
        # Residual refinement
        # ----------------------------------------------------

        residual = self.residual(
            x
        )

        residual = self.norm(
            residual
        )

        x = self.relu(
            x + residual
        )

        return x


# ============================================================
# Transformer Bottleneck (BiT-style)
# ============================================================

class TransformerBottleneck(nn.Module):
    """
    Lightweight self-attention refinement at the deepest
    feature level.

    At f4, spatial resolution is only 8x8 = 64 positions, so
    running full self-attention here is cheap even on a 4GB
    GPU — unlike applying a transformer at f1 (64x64 = 4096
    tokens), which would blow the VRAM budget at batch_size=1.

    This lets the model relate distant regions of the image
    (e.g. a demolished building on one side and new
    construction on the other) that a purely convolutional
    receptive field would struggle to connect.
    """

    def __init__(
        self,
        channels: int = 512,
        spatial_size: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.channels = channels
        self.spatial_size = spatial_size
        num_tokens = spatial_size * spatial_size

        # Learned positional embedding — one per spatial
        # position, since attention is otherwise
        # permutation-invariant and would lose location info.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, channels)
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):

        b, c, h, w = x.shape

        # ----------------------------------------------------
        # [B, C, H, W] -> [B, H*W, C] (tokens)
        # ----------------------------------------------------

        tokens = x.flatten(2).transpose(1, 2)

        tokens = tokens + self.pos_embed

        # ----------------------------------------------------
        # Self-attention refinement
        # ----------------------------------------------------

        tokens = self.transformer(tokens)

        tokens = self.norm(tokens)

        # ----------------------------------------------------
        # [B, H*W, C] -> [B, C, H, W]
        # ----------------------------------------------------

        out = tokens.transpose(1, 2).reshape(b, c, h, w)

        # Residual connection so the model can fall back to
        # the pure-CNN features if attention isn't helping
        # for a given input.
        return x + out


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(nn.Module):
    """
    Upsample and fuse multi-scale features.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.conv(
            x
        )


# ============================================================
# Change Detector V5
# ============================================================

class ChangeDetector(nn.Module):
    """
    GeoSenseAI V5 Change Detection Module.

    Input feature sizes:

        f1 -> [B, 64, 64, 64]
        f2 -> [B, 128, 32, 32]
        f3 -> [B, 256, 16, 16]
        f4 -> [B, 512, 8, 8]

    Output:

        [B, 1, 256, 256]
    """

    def __init__(self):
        super().__init__()

        # ====================================================
        # V5 Temporal Fusion
        # ====================================================

        self.fuse1 = TemporalFusionBlockV5(
            64
        )

        self.fuse2 = TemporalFusionBlockV5(
            128
        )

        self.fuse3 = TemporalFusionBlockV5(
            256
        )

        self.fuse4 = TemporalFusionBlockV5(
            512
        )

        # ====================================================
        # Transformer Bottleneck (BiT-style)
        #
        # f4 is 8x8 spatially -> only 64 tokens, cheap enough
        # for self-attention on a 4GB GPU at batch_size=1.
        # ====================================================

        self.bottleneck = TransformerBottleneck(
            channels=512,
            spatial_size=8,
            num_layers=2,
            num_heads=8,
            ff_dim=1024,
        )

        # ====================================================
        # Multi-scale Decoder
        # ====================================================

        self.decoder3 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.decoder2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.decoder1 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        # ====================================================
        # Prediction Head
        # ====================================================

        self.final_conv = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
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
                kernel_size=1,
            ),
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        before_features: List[torch.Tensor],
        after_features: List[torch.Tensor],
    ):

        # ====================================================
        # Temporal Fusion
        # ====================================================

        f1 = self.fuse1(
            before_features[0],
            after_features[0],
        )

        f2 = self.fuse2(
            before_features[1],
            after_features[1],
        )

        f3 = self.fuse3(
            before_features[2],
            after_features[2],
        )

        f4 = self.fuse4(
            before_features[3],
            after_features[3],
        )

        # ====================================================
        # Transformer Bottleneck
        # ====================================================

        f4 = self.bottleneck(
            f4
        )

        # ====================================================
        # Decoder
        # ====================================================

        x = self.decoder3(
            f4,
            f3,
        )

        x = self.decoder2(
            x,
            f2,
        )

        x = self.decoder1(
            x,
            f1,
        )

        # ====================================================
        # 64x64 -> 256x256
        # ====================================================

        x = F.interpolate(
            x,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        # ====================================================
        # Final Mask
        # ====================================================

        mask = self.final_conv(
            x
        )

        return mask