import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.helper import set_seed, get_device

set_seed(42)

device = get_device()

print("=" * 40)
print("GeoSenseAI Helper Test")
print("=" * 40)
print(f"Device : {device}")
print("Helper functions working successfully!")
print("=" * 40)