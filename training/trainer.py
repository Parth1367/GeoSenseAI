"""
GeoSenseAI Trainer
"""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from training.losses import BCEDiceLoss
from training.metrics import (
    iou_score,
    dice_score,
    precision_score,
    recall_score,
    f1_score,
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config
    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.device = config.DEVICE

        self.model.to(self.device)

        self.loss_fn = BCEDiceLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        self.best_loss = float("inf")

        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )