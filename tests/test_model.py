import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from models.vision_encoder import VisionEncoder
from models.change_detector import ChangeDetector

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

encoder = VisionEncoder().to(device)

detector = ChangeDetector().to(device)

before = torch.randn(
    1,
    3,
    256,
    256
).to(device)

after = torch.randn(
    1,
    3,
    256,
    256
).to(device)

before_features = encoder(before)
after_features = encoder(after)

mask = detector(
    before_features,
    after_features
)

print("=" * 60)
print("GeoSenseAI Change Detector")
print("=" * 60)
print("Output Mask Shape:", mask.shape)
print("=" * 60)