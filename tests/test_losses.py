import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from training.losses import BCEDiceLoss

loss_fn = BCEDiceLoss()

pred = torch.randn(2, 1, 256, 256)

target = torch.randint(
    0,
    2,
    (2, 1, 256, 256)
).float()

loss = loss_fn(pred, target)

print("=" * 60)
print("Loss Function Test")
print("=" * 60)
print("Loss :", loss.item())
print("=" * 60)