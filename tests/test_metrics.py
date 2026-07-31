import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from training.metrics import (
    iou_score,
    dice_score,
    precision_score,
    recall_score,
    f1_score,
)

pred = torch.randn(2, 1, 256, 256)

target = torch.randint(
    0,
    2,
    (2, 1, 256, 256)
).float()

print("=" * 60)
print("Metrics Test")
print("=" * 60)

print("IoU       :", iou_score(pred, target))
print("Dice      :", dice_score(pred, target))
print("Precision :", precision_score(pred, target))
print("Recall    :", recall_score(pred, target))
print("F1 Score  :", f1_score(pred, target))

print("=" * 60)