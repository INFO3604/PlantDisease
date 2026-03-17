#!/usr/bin/env python
"""Create train/val/test splits using stratified sampling.

Usage:
    python scripts/02_make_splits.py --manifest data/processed/manifest.csv --output data/splits
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease.data.splits import make_splits

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV")
    parser.add_argument("--output", type=Path, default=Path("data/splits"), help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Creating train/val/test splits...")
    make_splits(
        manifest_path=args.manifest,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    logger.info("Splits created successfully!")
