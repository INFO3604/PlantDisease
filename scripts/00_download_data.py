#!/usr/bin/env python
"""Download dataset from source and extract."""
import logging
from src.plantdisease.data.download import download_dataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Downloading dataset...")
    download_dataset()
    logger.info("Dataset download complete!")
