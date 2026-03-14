#!/usr/bin/env python
"""
CLI tool for evaluating trained models (XGBoost or CNN) on test sets.

This script evaluates both XGBoost and CNN classifiers and generates:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Per-class metrics
- Visualization plots

Usage examples:
    # Evaluate XGBoost on test features
    python scripts/evaluate_cli.py --model xgboost --model-path models/xgboost/xgboost_model \\
        --test-data data/features/test.npz
    
    # Evaluate CNN on image directory
    python scripts/evaluate_cli.py --model cnn --model-path models/cnn/best_model.pt \\
        --test-dir data/processed/test --img-size 224
    
    # Evaluate with detailed output and visualization
    python scripts/evaluate_cli.py --model cnn --model-path models/cnn/best_model.pt \\
        --test-dir data/processed/test --output reports/evaluation \\
        --plot --device cuda:0
    
    # Evaluate XGBoost with class names
    python scripts/evaluate_cli.py --model xgboost --model-path models/xgboost/xgboost_model \\
        --test-data data/features/test.npz --class-names healthy early_blight late_blight
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List
import pickle

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostEvaluator:
    """Evaluator for XGBoost models."""
    
    def __init__(self, model_path: Path):
        """Initialize evaluator with model."""
        self.model_path = Path(model_path)
        self.model = self._load_model()
    
    def _load_model(self):
        """Load XGBoost model."""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded XGBoost model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(
        self,
        test_features_path: Path,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on test features.
        
        Args:
            test_features_path: Path to test features NPZ file
            class_names: Optional list of class names
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        test_data = np.load(test_features_path, allow_pickle=True)
        X_test = test_data['features']
        y_test = test_data['labels']
        
        logger.info(f"Evaluating on {len(X_test)} test samples")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Compute metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        }
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            output_dict=True,
            zero_division=0
        )
        metrics['per_class_metrics'] = class_report
        
        logger.info("\nGlobal Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics


class CNNEvaluator:
    """Evaluator for CNN models."""
    
    def __init__(self, model_path: Path, device: str = 'cpu'):
        """Initialize evaluator with model."""
        import torch
        
        self.model_path = Path(model_path)
        self.device = device
        self.model = self._load_model()
        self.classes = None
    
    def _load_model(self):
        """Load CNN model."""
        try:
            import torch
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            logger.info(f"Loaded CNN model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_config(self, config_path: Optional[Path] = None):
        """Load model configuration."""
        if config_path is None:
            config_path = self.model_path.parent / 'config.json'
        
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            self.classes = config_dict.get('classes')
            logger.info(f"Loaded config from {config_path}")
            return config_dict
        
        return {}
    
    def evaluate(
        self,
        test_dir: Path,
        img_size: int = 224,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on test directory.
        
        Args:
            test_dir: Directory containing class subdirectories with images
            img_size: Image size for model input
            class_names: Optional list of class names
        
        Returns:
            Dictionary of evaluation metrics
        """
        import torch
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        from src.plantdisease.models.cnn_baseline import PlantDiseaseDataset
        
        # Load test dataset
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
        
        dataset = PlantDiseaseDataset(test_dir, transform=transform)
        
        if class_names is None:
            class_names = dataset.classes
        
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        logger.info(f"Evaluating on {len(dataset)} test samples")
        logger.info(f"Classes: {class_names}")
        
        # Evaluate
        all_preds = []
        all_labels = []
        all_proba = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                proba = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
        
        y_pred = np.array(all_preds)
        y_test = np.array(all_labels)
        y_pred_proba = np.array(all_proba)
        
        # Compute metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            output_dict=True,
            target_names=class_names,
            zero_division=0
        )
        metrics['per_class_metrics'] = class_report
        
        logger.info("\nGlobal Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics


def save_metrics(metrics: Dict, output_path: Path):
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_values(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_values(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    metrics_serializable = convert_values(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")


def print_metrics(metrics: Dict):
    """Print metrics in a readable format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("\nGlobal Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision (w):   {metrics['precision_weighted']:.4f}")
    print(f"  Precision (m):   {metrics['precision_macro']:.4f}")
    print(f"  Recall (w):      {metrics['recall_weighted']:.4f}")
    print(f"  Recall (m):      {metrics['recall_macro']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained plant disease models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate XGBoost model
  python scripts/evaluate_cli.py --model xgboost \\
      --model-path models/xgboost/xgboost_model \\
      --test-data data/features/test.npz
  
  # Evaluate CNN model
  python scripts/evaluate_cli.py --model cnn \\
      --model-path models/cnn/best_model.pt \\
      --test-dir data/processed/test --img-size 224
  
  # With custom output and device
  python scripts/evaluate_cli.py --model cnn \\
      --model-path models/cnn/best_model.pt \\
      --test-dir data/processed/test \\
      --output reports/evaluation --device cuda:0
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['xgboost', 'cnn'],
        required=True,
        help='Type of model to evaluate'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--test-data',
        type=Path,
        help='Path to test features NPZ file (for XGBoost)'
    )
    parser.add_argument(
        '--test-dir',
        type=Path,
        help='Test directory with class subdirectories (for CNN)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Image size for CNN input (default: 224)'
    )
    parser.add_argument(
        '--class-names',
        nargs='+',
        help='List of class names'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory for metrics'
    )
    parser.add_argument(
        '--device',
        default='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available, else cpu)'
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
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Device: {args.device}")
    
    # Validate paths
    if not args.model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        return 1
    
    try:
        if args.model == 'xgboost':
            if not args.test_data:
                logger.error("--test-data is required for XGBoost evaluation")
                return 1
            if not args.test_data.exists():
                logger.error(f"Test data not found: {args.test_data}")
                return 1
            
            evaluator = XGBoostEvaluator(args.model_path)
            metrics = evaluator.evaluate(args.test_data, args.class_names)
        
        elif args.model == 'cnn':
            if not args.test_dir:
                logger.error("--test-dir is required for CNN evaluation")
                return 1
            if not args.test_dir.exists():
                logger.error(f"Test directory not found: {args.test_dir}")
                return 1
            
            evaluator = CNNEvaluator(args.model_path, device=args.device)
            evaluator._load_config()
            metrics = evaluator.evaluate(
                args.test_dir,
                img_size=args.img_size,
                class_names=args.class_names or evaluator.classes
            )
        
        # Print results
        print_metrics(metrics)
        
        # Save metrics if output specified
        if args.output:
            save_metrics(metrics, Path(args.output) / 'evaluation_metrics.json')
        
        logger.info("Evaluation complete!")
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
