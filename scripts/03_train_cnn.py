#!/usr/bin/env python
"""Train CNN model (ResNet/EfficientNet) on plant disease dataset."""
import logging
from src.plantdisease.models.train import train_model

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting model training...")
    train_model()
    logger.info("Training complete!")
