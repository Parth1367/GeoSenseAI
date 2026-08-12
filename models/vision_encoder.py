"""
GeoSenseAI Vision Encoder

Purpose:
    Extract multi-scale feature maps from satellite images
    using a pretrained ResNet-18 backbone.

Author: Namdev
"""

from typing import List

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class VisionEncoder(nn.Module):
    """
    ResNet-18 based Vision Encoder.

    Input:
        Tensor -> [B, 3, H, W]

    Output:
        Multi-scale feature maps:

        f1 -> 64 channels
        f2 -> 128 channels
        f3 -> 256 channels
        f4 -> 512 channels
    """

    def __init__(self, pretrained: bool = True):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(weights=weights)

        # -----------------------------------------
        # Initial layers
        # -----------------------------------------

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # -----------------------------------------
        # Residual stages
        # -----------------------------------------

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:

        # Stem
        x = self.stem(x)

        # Multi-scale features
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [
            f1,
            f2,
            f3,
            f4,
        ]