#!/usr/bin/env python
"""
CLI tool for training Random Forest ensemble model for plant disease detection.

This script trains a Random Forest classifier on the 55-feature vector:
- 36 Gabor texture features (12 filter banks × 3 statistics)
- 6 CIELAB colour statistics (L*, a*, b* mean + std)
- 3 severity ratios (disease, yellow, brown)
- 10 morphological features (lesion count, area, perimeter, shape)

The features are expected to be pre-extracted and stored as NPZ files.

Usage examples:
    # Train with default parameters
    python scripts/train_rf_cli.py --train data/features/train.npz --val data/features/val.npz

    # Train with custom hyperparameters
    python scripts/train_rf_cli.py --train data/features/train.npz --val data/features/val.npz \\
        --n-estimators 500 --max-depth 20

    # Train without validation set
    python scripts/train_rf_cli.py --train data/features/train.npz --output models/rf_v2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease.models.rf_ensemble import train_rf_ensemble
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(npz_path: Path) -> tuple:
    """Load features from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    feature_names = (
        data["feature_names"].tolist() if "feature_names" in data else None
    )
    logger.info(f"Loaded features from {npz_path.name}")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Labels: {np.unique(labels)}")
    return features, labels, feature_names


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest ensemble for plant disease detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--train", type=Path, required=True, help="Training features NPZ")
    parser.add_argument("--val", type=Path, help="Validation features NPZ (optional)")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("models/rf_ensemble"),
        help="Output directory (default: models/rf_ensemble)",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees (default: 300)")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth (default: None=unlimited)")
    parser.add_argument("--min-samples-split", type=int, default=5, help="Min samples to split (default: 5)")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Min samples per leaf (default: 2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Random Forest Ensemble Training")
    logger.info("=" * 60)
    logger.info(f"Training features: {args.train}")
    logger.info(f"Validation features: {args.val or 'None'}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"  n_estimators: {args.n_estimators}")
    logger.info(f"  max_depth: {args.max_depth}")
    logger.info(f"  min_samples_split: {args.min_samples_split}")
    logger.info(f"  min_samples_leaf: {args.min_samples_leaf}")

    if not args.train.exists():
        logger.error(f"Training features file not found: {args.train}")
        return 1
    if args.val and not args.val.exists():
        logger.error(f"Validation features file not found: {args.val}")
        return 1

    try:
        classifier, metrics = train_rf_ensemble(
            train_features_path=args.train,
            val_features_path=args.val,
            output_dir=args.output,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )

        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {args.output}")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {name}: {value:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
