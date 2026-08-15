from pathlib import Path
from typing import Callable, Optional

import cv2
import torch
from torch.utils.data import Dataset


class LEVIRCDDataset(Dataset):
    """
    PyTorch Dataset for LEVIR-CD — patch-based.

    LEVIR-CD ships 1024x1024 image pairs. Instead of resizing
    each full image down to 256x256 (which throws away most of
    the spatial resolution and limits training data to just the
    number of image pairs), each 1024x1024 pair is split into
    16 non-overlapping 256x256 patches. This matches the
    standard LEVIR-CD preprocessing used in published benchmarks:

        445 train pairs  -> 7,120 train patches
        64  val pairs    -> 1,024 val patches
        128 test pairs   -> 2,048 test patches

    Returns:
        before_patch, after_patch, mask_patch
    """

    # ImageNet stats — required because VisionEncoder uses
    # ResNet18_Weights.DEFAULT (pretrained backbone)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    PATCH_SIZE = 256

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.a_dir = self.root_dir / split / "A"
        self.b_dir = self.root_dir / split / "B"
        self.label_dir = self.root_dir / split / "label"

        self.image_names = sorted(
            [img.name for img in self.a_dir.glob("*.png")]
        )

        self._verify_dataset()

        # Flat list of (image_name, row_offset, col_offset)
        # for every non-overlapping patch in every image.
        self.patches = self._build_patch_index()

        # In-memory cache for full (un-cropped) images, keyed
        # by filename. Without this, every one of the 16
        # patches per image re-reads and re-decodes the same
        # 1024x1024 PNG from disk — a 16x I/O waste. Since
        # DataLoader uses persistent_workers=True, this cache
        # survives across epochs too (first epoch pays full
        # I/O cost, later epochs mostly hit cache).
        self._cache = {}

    def _verify_dataset(self):

        for name in self.image_names:

            if not (self.b_dir / name).exists():
                raise FileNotFoundError(f"Missing file: {self.b_dir / name}")

            if not (self.label_dir / name).exists():
                raise FileNotFoundError(f"Missing file: {self.label_dir / name}")

    def _build_patch_index(self):
        """
        Reads one image to get its dimensions, then builds a
        list of non-overlapping patch coordinates for every
        image in this split. Assumes all images in a split
        share the same size (true for LEVIR-CD: 1024x1024).

        If an image's dimensions aren't an exact multiple of
        PATCH_SIZE, the remainder pixels at the bottom/right
        edge are dropped (LEVIR-CD's 1024x1024 divides evenly
        into 256x256 patches, so nothing is lost there).
        """

        patches = []

        if not self.image_names:
            return patches

        sample = cv2.imread(str(self.a_dir / self.image_names[0]))

        if sample is None:
            raise ValueError(
                f"Could not read sample image: "
                f"{self.a_dir / self.image_names[0]}"
            )

        height, width = sample.shape[:2]

        n_rows = height // self.PATCH_SIZE
        n_cols = width // self.PATCH_SIZE

        if n_rows == 0 or n_cols == 0:
            raise ValueError(
                f"Image size {height}x{width} is smaller than "
                f"PATCH_SIZE {self.PATCH_SIZE}. Cannot build patches."
            )

        for name in self.image_names:
            for row in range(n_rows):
                for col in range(n_cols):
                    patches.append(
                        (
                            name,
                            row * self.PATCH_SIZE,
                            col * self.PATCH_SIZE,
                        )
                    )

        return patches

    def __len__(self):

        return len(self.patches)

    def _load_full_images(self, filename):
        """
        Loads and decodes the full (un-cropped) before/after/
        mask images for a given filename, using the cache if
        already loaded.
        """

        if filename in self._cache:
            return self._cache[filename]

        before = cv2.imread(str(self.a_dir / filename))
        after = cv2.imread(str(self.b_dir / filename))
        mask = cv2.imread(str(self.label_dir / filename), cv2.IMREAD_GRAYSCALE)

        if before is None or after is None or mask is None:
            raise ValueError(
                f"Could not read one or more images for: {filename}"
            )

        before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

        self._cache[filename] = (before, after, mask)

        return before, after, mask

    def __getitem__(self, idx):

        filename, row, col = self.patches[idx]
        p = self.PATCH_SIZE

        before, after, mask = self._load_full_images(filename)

        # ------------------------------------------------
        # Crop to this patch's region (no resize — full
        # native resolution is preserved)
        # ------------------------------------------------

        before = before[row:row + p, col:col + p]
        after = after[row:row + p, col:col + p]
        mask = mask[row:row + p, col:col + p]

        before = torch.from_numpy(before.copy()).permute(2, 0, 1).float() / 255.0
        after = torch.from_numpy(after.copy()).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).float() / 255.0

        # Normalize with ImageNet stats (mask stays [0,1], untouched)
        before = (before - self.MEAN) / self.STD
        after = (after - self.MEAN) / self.STD

        if self.transform:
            before, after, mask = self.transform(before, after, mask)

        return before, after, mask