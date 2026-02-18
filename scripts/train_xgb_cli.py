#!/usr/bin/env python
"""
CLI tool for training XGBoost baseline model for plant disease detection.

This script trains an XGBoost classifier on hand-crafted features:
- HSV color histograms (96 dims)
- LBP texture features (26 dims)
- GLCM texture features (72 dims)

The features are expected to be pre-extracted and stored as NPZ files.

Usage examples:
    # Train with default parameters
    python scripts/train_xgb_cli.py --train data/features/train.npz --val data/features/val.npz
    
    # Train with custom hyperparameters
    python scripts/train_xgb_cli.py --train data/features/train.npz --val data/features/val.npz \\
        --n-estimators 200 --max-depth 8 --learning-rate 0.05
    
    # Train without validation set
    python scripts/train_xgb_cli.py --train data/features/train.npz --output models/xgboost_v2
    
    # Enable GPU acceleration
    python scripts/train_xgb_cli.py --train data/features/train.npz --val data/features/val.npz --gpu
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.models.xgboost_baseline import train_xgboost, XGBoostClassifier
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(npz_path: Path) -> tuple:
    """
    Load features from NPZ file.
    
    Args:
        npz_path: Path to NPZ file containing features and labels
    
    Returns:
        Tuple of (features, labels, feature_names)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        features = data['features']
        labels = data['labels']
        feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
        
        logger.info(f"Loaded features from {npz_path.name}")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Labels: {np.unique(labels)}")
        
        return features, labels, feature_names
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        raise


def compute_class_weights(labels: np.ndarray) -> Dict[str, float]:
    """
    Compute class weights for imbalanced data.
    
    Args:
        labels: Array of labels
    
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = {}
    for label, count in zip(unique_labels, counts):
        weight = total / (len(unique_labels) * count)
        weights[int(label)] = weight
    
    logger.info("Computed class weights:")
    for label, weight in sorted(weights.items()):
        logger.info(f"  Class {label}: {weight:.4f}")
    
    return weights


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier for plant disease detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with validation
  python scripts/train_xgb_cli.py --train data/features/train.npz \\
      --val data/features/val.npz
  
  # Custom hyperparameters
  python scripts/train_xgb_cli.py --train data/features/train.npz \\
      --val data/features/val.npz --n-estimators 300 --max-depth 10
  
  # With automatic class weighting
  python scripts/train_xgb_cli.py --train data/features/train.npz \\
      --val data/features/val.npz --auto-weights
  
  # GPU acceleration
  python scripts/train_xgb_cli.py --train data/features/train.npz \\
      --val data/features/val.npz --gpu
        """
    )
    
    parser.add_argument(
        '--train',
        type=Path,
        required=True,
        help='Path to training features NPZ file'
    )
    parser.add_argument(
        '--val',
        type=Path,
        help='Path to validation features NPZ file (optional)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('models/xgboost'),
        help='Output directory for trained model (default: models/xgboost)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of boosting rounds (default: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Boosting learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Subsample ratio of training data (default: 0.8)'
    )
    parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=0.8,
        help='Subsample ratio of columns per tree (default: 0.8)'
    )
    parser.add_argument(
        '--auto-weights',
        action='store_true',
        help='Automatically compute class weights for imbalanced data'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration (requires XGBoost with GPU support)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("XGBoost Baseline Model Training")
    logger.info("=" * 60)
    logger.info(f"Training features: {args.train}")
    logger.info(f"Validation features: {args.val if args.val else 'None'}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  n_estimators: {args.n_estimators}")
    logger.info(f"  max_depth: {args.max_depth}")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  subsample: {args.subsample}")
    logger.info(f"  colsample_bytree: {args.colsample_bytree}")
    logger.info(f"GPU acceleration: {args.gpu}")
    
    # Validate paths
    if not args.train.exists():
        logger.error(f"Training features file not found: {args.train}")
        return 1
    
    if args.val and not args.val.exists():
        logger.error(f"Validation features file not found: {args.val}")
        return 1
    
    # Load training data
    X_train, y_train, feature_names = load_features(args.train)
    
    # Load validation data if provided
    X_val, y_val = None, None
    if args.val:
        X_val, y_val, _ = load_features(args.val)
    
    # Compute class weights if requested
    class_weights = None
    if args.auto_weights:
        logger.info("Computing class weights for training data...")
        class_weights = compute_class_weights(y_train)
    
    # Prepare model parameters
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'random_state': args.random_state,
        'use_gpu': args.gpu,
        'class_weights': class_weights
    }
    
    logger.info("=" * 60)
    logger.info("Training model...")
    logger.info("=" * 60)
    
    try:
        # Train model
        classifier, metrics = train_xgboost(
            train_features_path=args.train,
            val_features_path=args.val,
            output_dir=args.output,
            class_weights=class_weights,
            **{k: v for k, v in model_params.items() if k != 'class_weights'}
        )
        
        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {args.output}")
        logger.info("\nMetrics:")
        for metric_name, metric_value in metrics.items():
            if metric_name != 'confusion_matrix':
                logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("=" * 60)
        logger.info("Artifacts saved:")
        logger.info(f"  - Model: {args.output / 'xgboost_model'}")
        logger.info(f"  - Metrics: {args.output / 'metrics.json'}")
        logger.info(f"  - Feature importance: {args.output / 'feature_importance.json'}")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
