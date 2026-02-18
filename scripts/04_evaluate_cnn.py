"""
Evaluation script for CNN model.

Usage:
    python scripts/04_evaluate_cnn.py --model checkpoints/best_model.pt --data-dir data/test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN, PlantDiseaseDataset, CNNTrainer
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Path,
    device: str = 'cpu'
) -> tuple:
    """Load trained model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = PlantDiseaseCNN(
        num_classes=model_config.get('num_classes', 10),
        backbone=model_config.get('backbone', 'mobilenet_v3_small'),
        pretrained=False,
        uncertainty_threshold=model_config.get('uncertainty_threshold', 0.5)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    
    return model, model_config


def create_test_loader(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> tuple:
    """Create test data loader."""
    
    dataset = PlantDiseaseDataset(data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return data_loader, dataset.get_class_names()


def evaluate_model(
    model: PlantDiseaseCNN,
    test_loader: DataLoader,
    class_names: list,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    
    logger.info("Evaluating model on test set...")
    
    trainer = CNNTrainer(model=model, device=device)
    metrics = trainer.evaluate(test_loader, class_names=class_names)
    
    return metrics


def print_metrics(metrics: Dict[str, Any], class_names: list) -> None:
    """Print evaluation metrics."""
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  Uncertain Rate: {metrics['uncertain_rate']:.4f}")
    
    if 'classification_report' in metrics:
        logger.info(f"\nPer-Class Metrics:")
        for class_name in class_names:
            report = metrics['classification_report'][class_name]
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {report['precision']:.4f}")
            logger.info(f"    Recall:    {report['recall']:.4f}")
            logger.info(f"    F1-Score:  {report['f1']:.4f}")
    
    logger.info("\n" + "="*60)


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    class_names: list,
    output_path: Path
) -> None:
    """Plot and save confusion matrix."""
    
    cm = np.array(metrics['confusion_matrix'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model')
    parser.add_argument('--model', type=Path, required=True, help='Path to checkpoint')
    parser.add_argument('--data-dir', type=Path, required=True, help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu', help='Device')
    parser.add_argument('--output', type=Path, help='Save metrics to JSON file')
    parser.add_argument('--confusion-matrix', type=Path, help='Save confusion matrix plot')
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model, model_config = load_model(args.model, device=args.device)
    
    # Create test loader
    logger.info(f"Loading test data from {args.data_dir}")
    test_loader, class_names = create_test_loader(args.data_dir, args.batch_size)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, class_names, args.device)
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    # Save metrics
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            # Convert confusion matrix to list for JSON serialization
            metrics_to_save = metrics.copy()
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Metrics saved to {args.output}")
    
    # Plot confusion matrix
    if args.confusion_matrix:
        plot_confusion_matrix(metrics, class_names, args.confusion_matrix)


if __name__ == '__main__':
    main()
