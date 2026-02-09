"""Configuration management for paths and hyperparameters."""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data paths
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
EXPORT_DIR = MODEL_DIR / "exports"

# Report paths
REPORT_DIR = Path(os.getenv("REPORT_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORT_DIR / "figures"
METRICS_DIR = REPORT_DIR / "metrics"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
                   CHECKPOINT_DIR, EXPORT_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 50))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
DEVICE = os.getenv("DEVICE", "cuda")

# Data parameters
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", 0.8))
VAL_RATIO = float(os.getenv("VAL_RATIO", 0.1))
TEST_RATIO = float(os.getenv("TEST_RATIO", 0.1))

# Image preprocessing
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
