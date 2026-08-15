"""
GeoSenseAI V5 Augmentation

Joint spatial augmentation applied identically to
before, after, and mask so alignment is preserved.

Used only on the TRAIN split. Do NOT apply to val/test.
"""

import random

import torch


def joint_transform(before, after, mask):
    """
    Applies identical spatial augmentation to before/after/mask.

    Expects before/after to already be normalized
    (ImageNet mean/std) and mask to be in [0, 1].
    """

    # ------------------------------------------------
    # Horizontal flip
    # ------------------------------------------------

    if random.random() < 0.5:
        before = torch.flip(before, dims=[-1])
        after = torch.flip(after, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    # ------------------------------------------------
    # Vertical flip
    # ------------------------------------------------

    if random.random() < 0.5:
        before = torch.flip(before, dims=[-2])
        after = torch.flip(after, dims=[-2])
        mask = torch.flip(mask, dims=[-2])

    # ------------------------------------------------
    # 90-degree rotation (0, 90, 180, 270)
    # ------------------------------------------------

    k = random.choice([0, 1, 2, 3])

    if k > 0:
        before = torch.rot90(before, k, dims=[-2, -1])
        after = torch.rot90(after, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])

    return before, after, mask