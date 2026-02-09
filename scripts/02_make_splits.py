#!/usr/bin/env python
"""Create train/val/test splits using stratified sampling."""
import logging
from src.plantdisease.data.splits import make_splits

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating train/val/test splits...")
    make_splits()
    logger.info("Splits created successfully!")
