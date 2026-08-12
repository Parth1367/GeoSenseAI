"""
GeoSenseAI V2 Final Test Metrics

Evaluates the trained GeoSenseAI V2 model on the
official LEVIR-CD test split.

Metrics:
    IoU
    Dice / F1
    Precision
    Recall
    Accuracy
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

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

import training.config as config

from datasets.levir_cd import LEVIRCDDataset
from models.geosense_ai import GeoSenseAI


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("GeoSenseAI V2 FINAL TEST METRICS")
    print("=" * 60)

    print(
        f"Device : {config.DEVICE}"
    )

    print(
        f"Dataset: {config.DATASET_PATH}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    test_dataset = LEVIRCDDataset(
        root_dir=config.DATASET_PATH,
        split="test"
    )

    print(
        f"Test Samples: {len(test_dataset)}"
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

    # ========================================================
    # MODEL
    # ========================================================

    model = GeoSenseAI()

    model = model.to(
        config.DEVICE
    )

    # ========================================================
    # V2 CHECKPOINT
    # ========================================================

    checkpoint_path = (
        PROJECT_ROOT
        / "weights"
        / "v2"
        / "best_model.pth"
    )

    print()
    print(
        f"Checkpoint: {checkpoint_path}"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"\nV2 checkpoint not found:\n"
            f"{checkpoint_path}\n\n"
            f"Make sure V2 training completed."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    best_epoch = (
        checkpoint.get("epoch", 0) + 1
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        "✓ V2 Best Model Loaded"
    )

    # ========================================================
    # GLOBAL CONFUSION MATRIX
    # ========================================================

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    processed = 0

    # ========================================================
    # EVALUATION
    # ========================================================

    print()
    print("=" * 60)
    print("Running V2 Test Evaluation...")
    print("=" * 60)

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Testing"
        )

        for before, after, target in progress:

            # ------------------------------------------------
            # Move data to GPU
            # ------------------------------------------------

            before = before.to(
                config.DEVICE,
                non_blocking=True
            )

            after = after.to(
                config.DEVICE,
                non_blocking=True
            )

            target = target.to(
                config.DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            logits = model(
                before,
                after
            )

            probability = torch.sigmoid(
                logits
            )

            # ------------------------------------------------
            # Binary mask
            # ------------------------------------------------

            prediction = (
                probability >= 0.5
            ).float()

            target = (
                target >= 0.5
            ).float()

            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------

            tp = (
                (prediction == 1)
                & (target == 1)
            ).sum().item()

            tn = (
                (prediction == 0)
                & (target == 0)
            ).sum().item()

            fp = (
                (prediction == 1)
                & (target == 0)
            ).sum().item()

            fn = (
                (prediction == 0)
                & (target == 1)
            ).sum().item()

            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn

            processed += 1

    # ========================================================
    # PROCESSING CHECK
    # ========================================================

    print()
    print(
        f"Processed: {processed}"
    )

    if processed == 0:

        raise RuntimeError(
            "No test samples were processed."
        )

    # ========================================================
    # METRICS
    # ========================================================

    eps = 1e-7

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + false_negative
            + eps
        )
    )

    # --------------------------------------------------------
    # Dice / F1
    # --------------------------------------------------------

    dice = (
        2 * true_positive
        /
        (
            2 * true_positive
            + false_positive
            + false_negative
            + eps
        )
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + eps
        )
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        true_positive
        /
        (
            true_positive
            + false_negative
            + eps
        )
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = (
        true_positive
        + true_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + eps
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("GeoSenseAI V2 TEST RESULTS")
    print("=" * 60)

    print()
    print("Confusion Matrix:")

    print(
        f"True Positive : {true_positive}"
    )

    print(
        f"True Negative : {true_negative}"
    )

    print(
        f"False Positive: {false_positive}"
    )

    print(
        f"False Negative: {false_negative}"
    )

    print()
    print("Performance Metrics:")

    print(
        f"IoU        : {iou:.4f} "
        f"({iou * 100:.2f}%)"
    )

    print(
        f"Dice / F1  : {dice:.4f} "
        f"({dice * 100:.2f}%)"
    )

    print(
        f"Precision  : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall     : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Accuracy   : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print()
    print("=" * 60)
    print("✓ V2 FINAL METRICS SUCCESSFULLY CALCULATED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()