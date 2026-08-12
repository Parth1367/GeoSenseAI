"""
GeoSenseAI Official Test Evaluation
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import training.config as config

from datasets.levir_cd import LEVIRCDDataset
from models.geosense_ai import GeoSenseAI


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

    probability = torch.sigmoid(prediction)

    predicted = (probability >= 0.5).float()
    target = (target >= 0.5).float()

    predicted = predicted.reshape(-1)
    target = target.reshape(-1)

    tp = (predicted * target).sum()
    fp = (predicted * (1 - target)).sum()
    fn = ((1 - predicted) * target).sum()
    tn = ((1 - predicted) * (1 - target)).sum()

    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)

    dice = (
        2 * tp
        /
        (2 * tp + fp + fn + eps)
    )

    precision = (
        tp
        /
        (tp + fp + eps)
    )

    recall = (
        tp
        /
        (tp + fn + eps)
    )

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn + eps)
    )

    return (
        iou.item(),
        dice.item(),
        precision.item(),
        recall.item(),
        accuracy.item()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("GeoSenseAI Official Test Evaluation")
    print("=" * 60)

    print(
        f"Device  : {config.DEVICE}"
    )

    print(
        f"Dataset : {config.DATASET_PATH}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("Loading official test dataset...")

    test_dataset = LEVIRCDDataset(
        root_dir=config.DATASET_PATH,
        split="test"
    )

    print(
        f"Test Samples : {len(test_dataset)}"
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            config.DEVICE.type == "cuda"
        )
    )

    print("✓ Test DataLoader Ready")

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Loading GeoSenseAI model...")

    model = GeoSenseAI()

    model = model.to(config.DEVICE)

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = (
        PROJECT_ROOT
        / "weights"
        / "best_model.pth"
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Best model not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("✓ Best model loaded")

    if "epoch" in checkpoint:

        print(
            f"Checkpoint Epoch : "
            f"{checkpoint['epoch'] + 1}"
        )

    # ========================================================
    # METRIC ACCUMULATORS
    # ========================================================

    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_accuracy = 0.0

    processed = 0

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 60)
    print("Running Test Evaluation")
    print("=" * 60)

    with torch.no_grad():

        for before, after, mask in tqdm(
            test_loader,
            desc="Testing"
        ):

            before = before.to(
                config.DEVICE,
                non_blocking=True
            )

            after = after.to(
                config.DEVICE,
                non_blocking=True
            )

            mask = mask.to(
                config.DEVICE,
                non_blocking=True
            )

            prediction = model(
                before,
                after
            )

            (
                iou,
                dice,
                precision,
                recall,
                accuracy
            ) = calculate_metrics(
                prediction,
                mask
            )

            total_iou += iou
            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_accuracy += accuracy

            processed += 1

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print()
    print("=" * 60)
    print("TEST LOOP FINISHED")
    print("=" * 60)

    print(
        f"Processed Samples : {processed}"
    )

    if processed == 0:
        print("ERROR: No samples processed.")
        return

    avg_iou = total_iou / processed
    avg_dice = total_dice / processed
    avg_precision = total_precision / processed
    avg_recall = total_recall / processed
    avg_accuracy = total_accuracy / processed

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("GeoSenseAI TEST RESULTS")
    print("=" * 60)

    print(
        "IoU        : "
        f"{avg_iou:.4f} "
        f"({avg_iou * 100:.2f}%)"
    )

    print(
        "Dice / F1  : "
        f"{avg_dice:.4f} "
        f"({avg_dice * 100:.2f}%)"
    )

    print(
        "Precision  : "
        f"{avg_precision:.4f} "
        f"({avg_precision * 100:.2f}%)"
    )

    print(
        "Recall     : "
        f"{avg_recall:.4f} "
        f"({avg_recall * 100:.2f}%)"
    )

    print(
        "Accuracy   : "
        f"{avg_accuracy:.4f} "
        f"({avg_accuracy * 100:.2f}%)"
    )

    print("=" * 60)
    print("✓ METRICS CALCULATION SUCCESSFUL")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()