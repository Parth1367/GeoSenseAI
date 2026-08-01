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
        config,
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
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        self.best_loss = float("inf")

        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0

        progress = tqdm(self.train_loader, desc="Training", leave=False)

        for before, after, mask in progress:

            before = before.to(self.device, non_blocking=True)
            after = after.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            prediction = self.model(before, after)

            loss = self.loss_fn(prediction, mask)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            total_iou += iou_score(prediction.detach(), mask)
            total_dice += dice_score(prediction.detach(), mask)

            progress.set_postfix(loss=loss.item())

        n = len(self.train_loader)

        return {
            "loss": total_loss / n,
            "iou": total_iou / n,
            "dice": total_dice / n,
        }

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        progress = tqdm(self.val_loader, desc="Validation", leave=False)

        for before, after, mask in progress:

            before = before.to(self.device)
            after = after.to(self.device)
            mask = mask.to(self.device)

            prediction = self.model(before, after)

            loss = self.loss_fn(prediction, mask)

            total_loss += loss.item()

            total_iou += iou_score(prediction, mask)
            total_dice += dice_score(prediction, mask)
            total_precision += precision_score(prediction, mask)
            total_recall += recall_score(prediction, mask)
            total_f1 += f1_score(prediction, mask)

        n = len(self.val_loader)

        return {
            "loss": total_loss / n,
            "iou": total_iou / n,
            "dice": total_dice / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
            "f1": total_f1 / n,
        }

    def fit(self, num_epochs, checkpoint=None):

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for epoch in range(num_epochs):

            print("=" * 60)
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print("=" * 60)

            train_metrics = self.train_one_epoch()

            val_metrics = self.validate()

            self.scheduler.step(val_metrics["loss"])

            print(
                f"Train Loss : {train_metrics['loss']:.4f} | "
                f"IoU : {train_metrics['iou']:.4f} | "
                f"Dice : {train_metrics['dice']:.4f}"
            )

            print(
                f"Val Loss : {val_metrics['loss']:.4f} | "
                f"IoU : {val_metrics['iou']:.4f} | "
                f"Dice : {val_metrics['dice']:.4f} | "
                f"Precision : {val_metrics['precision']:.4f} | "
                f"Recall : {val_metrics['recall']:.4f} | "
                f"F1 : {val_metrics['f1']:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": val_metrics["loss"],
            }

            torch.save(
                checkpoint,
                self.checkpoint_dir / "last_model.pth",
            )

            if val_metrics["loss"] < self.best_loss:

                self.best_loss = val_metrics["loss"]

                torch.save(
                    checkpoint,
                    self.checkpoint_dir / "best_model.pth",
                )

                print("✅ Best model saved!")

        print("\n🎉 Training Completed!")