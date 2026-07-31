"""
GeoSenseAI
Complete End-to-End Change Detection Model
"""

import torch.nn as nn

from models.vision_encoder import VisionEncoder
from models.change_detector import ChangeDetector


class GeoSenseAI(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = VisionEncoder()

        self.detector = ChangeDetector()

    def forward(self, before, after):

        before_features = self.encoder(before)

        after_features = self.encoder(after)

        mask = self.detector(
            before_features,
            after_features
        )

        return mask