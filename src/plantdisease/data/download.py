"""Dataset download and extraction utilities."""
import os
import logging
from pathlib import Path
from src.plantdisease import config

logger = logging.getLogger(__name__)

def download_dataset(url=None, destination=None):
    """
    Download dataset from URL and extract.
    
    Args:
        url: Dataset URL (defaults to config.DATASET_URL)
        destination: Where to save (defaults to raw data directory)
    """
    if url is None:
        url = os.getenv("DATASET_URL")
    if destination is None:
        destination = config.RAW_DATA_DIR
    
    destination.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from {url} to {destination}")
    # TODO: Implement download and extraction logic
    #   - Use urllib, requests, or gdown
    #   - Handle ZIP/TAR files
    #   - Create manifest.json
    pass

def get_manifest(path=None):
    """
    Load dataset manifest (JSON with file lists and metadata).
    
    Args:
        path: Path to manifest file
    
    Returns:
        Dictionary with manifest data
    """
    if path is None:
        path = config.RAW_DATA_DIR / "manifest.json"
    
    # TODO: Load and return manifest
    pass
