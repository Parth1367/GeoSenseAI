from pathlib import Path
from typing import Callable, Optional

import cv2
import torch
from torch.utils.data import Dataset


class LEVIRCDDataset(Dataset):
    """
    PyTorch Dataset for LEVIR-CD.

    Returns:
        before_image, after_image, change_mask
    """

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

    def _verify_dataset(self):

        for name in self.image_names:

            if not (self.b_dir / name).exists():
                raise FileNotFoundError(f"Missing file: {self.b_dir / name}")

            if not (self.label_dir / name).exists():
                raise FileNotFoundError(f"Missing file: {self.label_dir / name}")

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, idx):

        filename = self.image_names[idx]

        before = cv2.imread(str(self.a_dir / filename))
        after = cv2.imread(str(self.b_dir / filename))
        mask = cv2.imread(str(self.label_dir / filename), cv2.IMREAD_GRAYSCALE)

        before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

        before = cv2.resize(before, (256, 256))
        after = cv2.resize(after, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        before = torch.from_numpy(before).permute(2, 0, 1).float() / 255.0
        after = torch.from_numpy(after).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        if self.transform:
            before, after, mask = self.transform(before, after, mask)

        return before, after, mask