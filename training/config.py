"""
GeoSenseAI Training Configuration
"""

from pathlib import Path
import torch

# =====================================================
# DATASET
# =====================================================

# Root folder of LEVIR-CD dataset
# Structure:
# data/
#   levir_cd/
#       train/
#       val/
#       test/
#
DATASET_PATH = r"C:\GeoSenseAI\data\levir_cd"

# =====================================================
# IMAGE
# =====================================================

IMAGE_SIZE = 256

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 1          # GTX 1650 4GB ke liye best
NUM_EPOCHS = 30         # Testing ke liye (baad me 50 karenge)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 2

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# OUTPUTS
# =====================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PREDICTION_DIR = OUTPUT_DIR / "predictions"
PREDICTION_DIR.mkdir(exist_ok=True)

# =====================================================
# CHECKPOINTS
# =====================================================

CHECKPOINT_DIR = Path("weights")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL = CHECKPOINT_DIR / "last_model.pth"

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42

# =====================================================
# PRINT CONFIG
# =====================================================

