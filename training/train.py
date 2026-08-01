"""
GeoSenseAI Training Script
"""

import sys
from pathlib import Path

import torch
torch.cuda.empty_cache()

sys.path.append(str(Path(__file__).resolve().parents[1]))

import training.config as config
from torch.utils.data import DataLoader

from datasets.levir_cd import LEVIRCDDataset
from models.geosense_ai import GeoSenseAI
from training.trainer import Trainer


def main():

    print("=" * 60)
    print("GeoSenseAI Training")
    print("=" * 60)

    print(f"Device : {config.DEVICE}")

    # -----------------------------
    # Dataset
    # -----------------------------

    print("\nLoading datasets...")

    train_dataset = LEVIRCDDataset(
        root_dir=config.DATASET_PATH,
        split="train"
    )

    # Change "val" to "test" if your dataset doesn't contain a val folder.
    val_dataset = LEVIRCDDataset(
        root_dir=config.DATASET_PATH,
        split="val"
    )

    print(f"Train Samples : {len(train_dataset)}")
    print(f"Val Samples   : {len(val_dataset)}")

    # -----------------------------
    # DataLoader
    # -----------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0
    )

    print("✓ DataLoaders Ready")

    # -----------------------------
    # Model
    # -----------------------------

    print("Loading GeoSenseAI Model...")

    model = GeoSenseAI()

    print("✓ Model Loaded")

    # -----------------------------
    # Trainer
    # -----------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    print("✓ Trainer Ready")

    print("\nStarting Training...\n")

    trainer.fit(config.NUM_EPOCHS)

    print("\nTraining Finished Successfully!")


if __name__ == "__main__":
    main()