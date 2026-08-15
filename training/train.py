"""
GeoSenseAI V2 Training Script
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# CUDA MEMORY
# ============================================================

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# PROJECT IMPORTS
# ============================================================

import training.config as config

from datasets.levir_cd import LEVIRCDDataset
from datasets.transforms import joint_transform
from models.geosense_ai import GeoSenseAI
from training.trainer import Trainer


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("GeoSenseAI V2 Training")
    print("=" * 60)

    print(
        f"Device        : {config.DEVICE}"
    )

    print(
        f"Batch Size    : {config.BATCH_SIZE}"
    )

    print(
        f"Epochs        : {config.NUM_EPOCHS}"
    )

    print(
        f"Learning Rate : {config.LEARNING_RATE}"
    )

    print(
        f"AMP Enabled   : {config.USE_AMP}"
    )

    print(
        f"Checkpoint    : {config.CHECKPOINT_DIR}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("Loading datasets...")

    train_dataset = LEVIRCDDataset(
        root_dir=config.DATASET_PATH,
        split="train",
        transform=joint_transform
    )

    val_dataset = LEVIRCDDataset(
        root_dir=config.DATASET_PATH,
        split="val"
    )

    print(
        f"Train Samples : {len(train_dataset)}"
    )

    print(
        f"Val Samples   : {len(val_dataset)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    pin_memory = (
        config.DEVICE.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS,

        pin_memory=pin_memory,

        persistent_workers=(
            config.NUM_WORKERS > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=pin_memory,

        persistent_workers=(
            config.NUM_WORKERS > 0
        )
    )

    print(
        "✓ DataLoaders Ready"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Loading GeoSenseAI V2 Model...")

    model = GeoSenseAI()

    print(
        "✓ V2 Model Loaded"
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        config=config
    )

    print(
        "✓ Trainer Ready"
    )

    # ========================================================
    # RESUME (if a previous checkpoint exists)
    # ========================================================

    checkpoint = None

    last_checkpoint_path = (
        config.CHECKPOINT_DIR
        / "last_model.pth"
    )

    if last_checkpoint_path.exists():

        print()
        print(
            f"Found existing checkpoint: {last_checkpoint_path}"
        )

        print(
            "Resuming from this checkpoint "
            "(model + optimizer weights restored)..."
        )

        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=config.DEVICE
        )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 60)
    print("Starting V2 Training...")
    print("=" * 60)

    trainer.fit(
        config.NUM_EPOCHS,
        checkpoint=checkpoint,
        # Previous run's val-loss NaN bug collapsed the LR to
        # ~1e-6 via a corrupted scheduler. Reset it to a
        # healthy value when resuming so training can actually
        # keep improving now that the bug is fixed.
        resume_lr=3e-5 if checkpoint is not None else None,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 60)
    print("V2 Training Finished Successfully!")
    print("=" * 60)

    print(
        f"Best model saved at:"
    )

    print(
        config.BEST_MODEL
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()