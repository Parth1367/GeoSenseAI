import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.logger import get_logger

logger = get_logger()

logger.info("GeoSenseAI Logger Started")
logger.warning("This is a warning message.")
logger.error("This is a sample error message.")

print("Logger Test Completed Successfully!")