"""
GeoSenseAI V3 Inference Script

Usage:
    python predict.py --before before.png --after after.png

Outputs:
    outputs/predictions/prediction.png
    outputs/predictions/overlay.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

import training.config as config

from models.geosense_ai import GeoSenseAI


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.30

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Resize to model input
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    image_tensor = (
        torch.from_numpy(
            image
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return image, image_tensor


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = GeoSenseAI()

    model = model.to(
        config.DEVICE
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "weights"
        / "v3"
        / "best_model.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"V3 checkpoint not found:\n"
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

    return model


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    mask,
    output_path
):

    mask_uint8 = (
        (mask * 255)
        .astype(np.uint8)
    )

    cv2.imwrite(
        str(output_path),
        mask_uint8
    )


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_overlay(
    image,
    mask,
    output_path
):

    base = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Red overlay for detected change
    overlay = base.copy()

    overlay[
        mask == 1
    ] = [0, 0, 255]

    blended = cv2.addWeighted(
        base,
        0.70,
        overlay,
        0.30,
        0
    )

    cv2.imwrite(
        str(output_path),
        blended
    )


# ============================================================
# MAIN PREDICTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GeoSenseAI V3 Change Detection"
    )

    parser.add_argument(
        "--before",
        required=True,
        help="Path to before satellite image"
    )

    parser.add_argument(
        "--after",
        required=True,
        help="Path to after satellite image"
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("GeoSenseAI V3 Inference")
    print("=" * 60)

    print(
        f"Device    : {config.DEVICE}"
    )

    print(
        f"Threshold : {THRESHOLD}"
    )

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    print()
    print("Loading images...")

    before_display, before = load_image(
        args.before
    )

    after_display, after = load_image(
        args.after
    )

    print("✓ Images loaded")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading V3 model...")

    model = load_model()

    print("✓ V3 model loaded")

    # ========================================================
    # INFERENCE
    # ========================================================

    before = before.unsqueeze(
        0
    ).to(config.DEVICE)

    after = after.unsqueeze(
        0
    ).to(config.DEVICE)

    print()
    print("Running change detection...")

    with torch.no_grad():

        logits = model(
            before,
            after
        )

        probability = torch.sigmoid(
            logits
        )

        prediction = (
            probability >= THRESHOLD
        )

    mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    probability_map = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    print("✓ Prediction completed")

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    prediction_path = (
        OUTPUT_DIR
        / "prediction.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / "overlay.png"
    )

    probability_path = (
        OUTPUT_DIR
        / "probability.png"
    )

    # ========================================================
    # SAVE MASK
    # ========================================================

    save_mask(
        mask,
        prediction_path
    )

    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    save_overlay(
        after_display,
        mask,
        overlay_path
    )

    # ========================================================
    # SAVE PROBABILITY MAP
    # ========================================================

    probability_uint8 = (
        probability_map * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(probability_path),
        probability_uint8
    )

    # ========================================================
    # RESULTS
    # ========================================================

    changed_pixels = int(
        mask.sum()
    )

    total_pixels = mask.size

    change_percentage = (
        changed_pixels
        / total_pixels
        * 100
    )

    print()
    print("=" * 60)
    print("Prediction Completed")
    print("=" * 60)

    print(
        f"Changed Pixels : {changed_pixels}"
    )

    print(
        f"Change Area    : "
        f"{change_percentage:.2f}%"
    )

    print()
    print(
        f"Prediction : {prediction_path}"
    )

    print(
        f"Overlay    : {overlay_path}"
    )

    print(
        f"Probability: {probability_path}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()