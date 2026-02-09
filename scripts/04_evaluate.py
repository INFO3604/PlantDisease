#!/usr/bin/env python
"""Evaluate trained model on test set."""
import logging
from src.plantdisease.models.evaluate import evaluate_model

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluating model...")
    evaluate_model()
    logger.info("Evaluation complete!")
