import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from models.geosense_ai import GeoSenseAI

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = GeoSenseAI().to(device)

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

output = model(
    before,
    after
)

print("=" * 60)
print("GeoSenseAI Full Model Test")
print("=" * 60)
print("Output Shape :", output.shape)
print("=" * 60)