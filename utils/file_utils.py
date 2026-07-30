"""
File utilities for GeoSenseAI.
"""

from pathlib import Path
import torch


def create_directory(path: str):
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, filepath: str):
    """
    Save model weights.
    """
    create_directory(Path(filepath).parent)
    torch.save(model.state_dict(), filepath)


def load_checkpoint(model, filepath: str, device="cpu"):
    """
    Load model weights.
    """
    model.load_state_dict(
        torch.load(filepath, map_location=device)
    )
    return model


def file_exists(filepath: str):
    """
    Check whether a file exists.
    """
    return Path(filepath).exists()