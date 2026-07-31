"""
GeoSenseAI Training Configuration
"""

import torch
from pathlib import Path

# ==========================
# Dataset
# ==========================

DATASET_PATH = "data/levir_cd"

# ==========================
# Training
# ==========================

IMAGE_SIZE = 256

BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================
# Checkpoints
# ==========================

CHECKPOINT_DIR = Path("weights")

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"

LAST_MODEL = CHECKPOINT_DIR / "last_model.pth"