"""
GeoSenseAI Trainer

Features:
    - GPU / CPU support
    - Automatic Mixed Precision (AMP) on CUDA
    - Gradient scaling
    - Validation metrics
    - Learning-rate scheduling
    - Checkpoint saving
    - Epoch timing
"""

from pathlib import Path
import time

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

        # =================================================
        # Loss
        # =================================================

        self.loss_fn = BCEDiceLoss()

        # =================================================
        # Optimizer
        # =================================================

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # =================================================
        # Scheduler
        # =================================================

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        # =================================================
        # AMP
        # =================================================

        self.use_amp = self.device.type == "cuda"

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        # =================================================
        # Checkpoints
        # =================================================

        self.best_loss = float("inf")

        self.checkpoint_dir = Path(
            config.CHECKPOINT_DIR
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"AMP Enabled : {self.use_amp}"
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0

        start_time = time.time()

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False
        )

        for before, after, mask in progress:

            before = before.to(
                self.device,
                non_blocking=True
            )

            after = after.to(
                self.device,
                non_blocking=True
            )

            mask = mask.to(
                self.device,
                non_blocking=True
            )

            # ---------------------------------------------
            # Clear gradients
            # ---------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------
            # Forward + AMP
            # ---------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):

                prediction = self.model(
                    before,
                    after
                )

            # Loss computed in fp32 — DiceLoss involves sigmoid
            # and division, which can overflow/underflow under
            # fp16 and silently produce NaN. BCEWithLogitsLoss
            # is fine in fp16, but Dice isn't, so the whole
            # loss computation is cast up to fp32 for safety.
            loss = self.loss_fn(
                prediction.float(),
                mask
            )

            # ---------------------------------------------
            # Backward
            # ---------------------------------------------

            if self.use_amp:

                self.scaler.scale(
                    loss
                ).backward()

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                loss.backward()

                self.optimizer.step()

            # ---------------------------------------------
            # Metrics
            # ---------------------------------------------

            total_loss += loss.item()

            total_iou += iou_score(
                prediction.detach(),
                mask
            )

            total_dice += dice_score(
                prediction.detach(),
                mask
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_time = time.time() - start_time

        n = len(self.train_loader)

        return {
            "loss": total_loss / n,
            "iou": total_iou / n,
            "dice": total_dice / n,
            "time": epoch_time,
        }

    # =====================================================
    # VALIDATION
    # =====================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False
        )

        for before, after, mask in progress:

            before = before.to(
                self.device,
                non_blocking=True
            )

            after = after.to(
                self.device,
                non_blocking=True
            )

            mask = mask.to(
                self.device,
                non_blocking=True
            )

            # ---------------------------------------------
            # AMP inference
            # ---------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):

                prediction = self.model(
                    before,
                    after
                )

            # Same fp32 cast as training — this is the fix for
            # val loss showing up as NaN.
            loss = self.loss_fn(
                prediction.float(),
                mask
            )

            # ---------------------------------------------
            # Metrics
            # ---------------------------------------------

            total_loss += loss.item()

            total_iou += iou_score(
                prediction,
                mask
            )

            total_dice += dice_score(
                prediction,
                mask
            )

            total_precision += precision_score(
                prediction,
                mask
            )

            total_recall += recall_score(
                prediction,
                mask
            )

            total_f1 += f1_score(
                prediction,
                mask
            )

        n = len(self.val_loader)

        return {
            "loss": total_loss / n,
            "iou": total_iou / n,
            "dice": total_dice / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
            "f1": total_f1 / n,
        }

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        num_epochs,
        checkpoint=None,
        resume_lr=None,
    ):

        # -------------------------------------------------
        # Resume checkpoint
        # -------------------------------------------------

        if checkpoint is not None:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            print(
                "✓ Checkpoint loaded"
            )

            # ---------------------------------------------
            # Optional LR reset on resume.
            #
            # If a previous run's scheduler collapsed the LR
            # (e.g. due to a NaN val loss bug always failing
            # the "improved" check), the optimizer state we
            # just loaded carries that collapsed LR forward.
            # This lets us override it explicitly.
            # ---------------------------------------------

            if resume_lr is not None:

                for group in self.optimizer.param_groups:
                    group["lr"] = resume_lr

                print(
                    f"✓ Learning rate reset to {resume_lr}"
                )

        # -------------------------------------------------
        # Training loop
        # -------------------------------------------------

        for epoch in range(num_epochs):

            print()
            print("=" * 60)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}]"
            )
            print("=" * 60)

            # ---------------------------------------------
            # Training
            # ---------------------------------------------

            train_metrics = (
                self.train_one_epoch()
            )

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            val_metrics = self.validate()

            # ---------------------------------------------
            # Scheduler
            #
            # Guard against NaN val loss ever corrupting the
            # schedule again — NaN comparisons are always
            # False, which previously caused the scheduler to
            # think "no improvement" every single epoch and
            # collapse the LR. If this ever happens again, skip
            # the step instead of feeding it garbage.
            # ---------------------------------------------

            val_loss = val_metrics["loss"]

            if val_loss == val_loss:  # False only when NaN

                self.scheduler.step(
                    val_loss
                )

            else:

                print(
                    "⚠️  Val loss is NaN this epoch — "
                    "skipping scheduler step to avoid "
                    "corrupting the learning rate."
                )

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # ---------------------------------------------
            # Results
            # ---------------------------------------------

            print()

            print(
                f"Train Loss : "
                f"{train_metrics['loss']:.4f} | "
                f"IoU : "
                f"{train_metrics['iou']:.4f} | "
                f"Dice : "
                f"{train_metrics['dice']:.4f}"
            )

            print(
                f"Val Loss : "
                f"{val_metrics['loss']:.4f} | "
                f"IoU : "
                f"{val_metrics['iou']:.4f} | "
                f"Dice : "
                f"{val_metrics['dice']:.4f} | "
                f"Precision : "
                f"{val_metrics['precision']:.4f} | "
                f"Recall : "
                f"{val_metrics['recall']:.4f} | "
                f"F1 : "
                f"{val_metrics['f1']:.4f}"
            )

            print(
                f"Learning Rate : {current_lr:.6f}"
            )

            print(
                f"Epoch Time : "
                f"{train_metrics['time'] / 60:.2f} minutes"
            )

            # ---------------------------------------------
            # Checkpoint
            # ---------------------------------------------

            checkpoint_data = {

                "epoch": epoch,

                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "scheduler_state_dict":
                    self.scheduler.state_dict(),

                "loss":
                    val_metrics["loss"],

            }

            # Last model

            torch.save(
                checkpoint_data,
                self.checkpoint_dir
                / "last_model.pth"
            )

            # Best model

            if (
                val_metrics["loss"]
                < self.best_loss
            ):

                self.best_loss = (
                    val_metrics["loss"]
                )

                torch.save(
                    checkpoint_data,
                    self.checkpoint_dir
                    / "best_model.pth"
                )

                print(
                    "✅ Best model saved!"
                )

        print()
        print("=" * 60)
        print("🎉 Training Completed!")
        print("=" * 60)