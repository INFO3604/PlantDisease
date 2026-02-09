"""Path utilities for data and model directories."""
from pathlib import Path
from src.plantdisease import config

def get_data_dir(subdirectory=None):
    """Get data directory path."""
    path = config.DATA_DIR
    if subdirectory:
        path = path / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_model_dir(subdirectory=None):
    """Get model directory path."""
    path = config.MODEL_DIR
    if subdirectory:
        path = path / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_report_dir(subdirectory=None):
    """Get report directory path."""
    path = config.REPORT_DIR
    if subdirectory:
        path = path / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path
