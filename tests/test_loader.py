import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.levir_cd import LEVIRCDDataset
from utils.visualize import visualize_sample

dataset = LEVIRCDDataset(
    root_dir="data/levir_cd",
    split="train"
)

print("Dataset Size:", len(dataset))

before, after, mask = dataset[0]

print(before.shape)
print(after.shape)
print(mask.shape)

visualize_sample(before, after, mask)