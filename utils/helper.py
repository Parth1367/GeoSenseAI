"""
Common helper functions for GeoSenseAI.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Return available device.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def move_to_device(*items, device):
    """
    Move tensors to selected device.
    """

    return [item.to(device) for item in items]