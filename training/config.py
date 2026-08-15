"""
GeoSenseAI V5 Training Configuration
"""

from pathlib import Path

import torch


# ============================================================
# DATASET
# ============================================================

# Root folder of LEVIR-CD dataset
#
# Structure:
#
# data/
#   levir_cd/
#       train/
#       val/
#       test/

DATASET_PATH = r"C:\GeoSenseAI\data\levir_cd"


# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 256


# ============================================================
# TRAINING
# ============================================================

# GTX 1650 4GB
BATCH_SIZE = 1

# NOTE: train.py now auto-resumes from weights/v5/last_model.pth
# if it exists. This number is "how many MORE epochs to run
# from wherever the last checkpoint left off" — NOT total
# epochs since the beginning. Since 3 epochs are already done,
# setting this to 30 means ~33 total epochs of training.
#
# At ~32 min/epoch, 30 epochs ≈ 16 hours. Run this overnight
# or during a long block of uninterrupted time.
NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MIXED PRECISION
# ============================================================

# AMP helps reduce GPU memory usage
# and can improve training speed.

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = Path(
    "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PREDICTION_DIR = (
    OUTPUT_DIR
    / "predictions"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# V5 CHECKPOINTS
# ============================================================

# IMPORTANT:
# V1, V2, V3, V4 checkpoints are NOT touched.
#
# V3 (current best, do not overwrite):
#   weights/v3/best_model.pth
#   weights/v3/last_model.pth
#
# V5:
#   weights/v5/best_model.pth
#   weights/v5/last_model.pth

CHECKPOINT_DIR = (
    Path("weights")
    / "v5"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_MODEL = (
    CHECKPOINT_DIR
    / "best_model.pth"
)


LAST_MODEL = (
    CHECKPOINT_DIR
    / "last_model.pth"
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 60)
print("GeoSenseAI V5 Configuration")
print("=" * 60)

print(
    f"Device        : {DEVICE}"
)

print(
    f"Dataset Path  : {DATASET_PATH}"
)

print(
    f"Batch Size    : {BATCH_SIZE}"
)

print(
    f"Epochs        : {NUM_EPOCHS}"
)

print(
    f"Learning Rate : {LEARNING_RATE}"
)

print(
    f"AMP Enabled   : {USE_AMP}"
)

print(
    f"Checkpoint    : {CHECKPOINT_DIR}"
)

print("=" * 60)