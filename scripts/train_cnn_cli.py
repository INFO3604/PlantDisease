#!/usr/bin/env python
"""
CLI tool for training CNN baseline model for plant disease detection.

This script trains a CNN classifier using transfer learning from ImageNet:
- MobileNetV3 Small/Large (lightweight for mobile deployment)
- EfficientNet-B0 (balance of accuracy and efficiency)

Supports data augmentation, learning rate scheduling, and model checkpointing.
Exports trained models in TorchScript and ONNX formats for mobile deployment.

Usage examples:
    # Train MobileNetV3-Small from directory structure
    python scripts/train_cnn_cli.py --data-dir data/processed --backbone mobilenet_v3_small
    
    # Train with custom hyperparameters
    python scripts/train_cnn_cli.py --data-dir data/processed --backbone efficientnet_b0 \\
        --epochs 100 --batch-size 64 --learning-rate 0.001 --img-size 256
    
    # With data augmentation and learning rate scheduling
    python scripts/train_cnn_cli.py --data-dir data/processed --augment --scheduler cosine \\
        --epochs 50
    
    # Resume from checkpoint
    python scripts/train_cnn_cli.py --data-dir data/processed --checkpoint models/checkpoints/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.models.cnn_baseline import (
    PlantDiseaseCNN, PlantDiseaseDataset, CNNTrainer
)
from src.plantdisease.utils.logger import get_logger
import torchvision.transforms as transforms

logger = get_logger(__name__)


def create_transforms(
    img_size: int = 224,
    augment: bool = False
) -> dict:
    """
    Create image transforms for training and validation.
    
    Args:
        img_size: Image size
        augment: Whether to apply data augmentation
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return {'train': train_transform, 'val': val_transform}


def split_dataset(
    dataset: PlantDiseaseDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[PlantDiseaseDataset, PlantDiseaseDataset, PlantDiseaseDataset]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import random_split
    
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    logger.info(f"Splitting dataset ({total} samples):")
    logger.info(f"  Train: {train_size} ({train_ratio*100:.1f}%)")
    logger.info(f"  Val:   {val_size} ({val_ratio*100:.1f}%)")
    logger.info(f"  Test:  {test_size} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train CNN model for plant disease detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train_cnn_cli.py --data-dir data/processed
  
  # MobileNetV3 with augmentation
  python scripts/train_cnn_cli.py --data-dir data/processed \\
      --backbone mobilenet_v3_small --augment --epochs 50
  
  # EfficientNet with custom hyperparameters
  python scripts/train_cnn_cli.py --data-dir data/processed \\
      --backbone efficientnet_b0 --epochs 100 --batch-size 64 \\
      --learning-rate 0.001 --img-size 256
  
  # Resume from checkpoint
  python scripts/train_cnn_cli.py --data-dir data/processed \\
      --checkpoint models/checkpoints/best.pt
  
  # With GPU
  python scripts/train_cnn_cli.py --data-dir data/processed \\
      --device cuda:0
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing class subdirectories with images'
    )
    parser.add_argument(
        '--backbone',
        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0'],
        default='mobilenet_v3_small',
        help='Backbone architecture (default: mobilenet_v3_small)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation'
    )
    parser.add_argument(
        '--scheduler',
        choices=['cosine', 'step', 'plateau', 'none'],
        default='cosine',
        help='Learning rate scheduler (default: cosine)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('models/cnn'),
        help='Output directory for trained model (default: models/cnn)'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for training (default: cuda if available, else cpu)'
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
    logger.info("CNN Baseline Model Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Image size: {args.img_size}x{args.img_size}")
    logger.info(f"Training hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Scheduler: {args.scheduler}")
    logger.info(f"Data augmentation: {args.augment}")
    logger.info(f"Device: {args.device}")
    
    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        logger.info("=" * 60)
        logger.info("Loading dataset...")
        logger.info("=" * 60)
        
        transforms_dict = create_transforms(args.img_size, args.augment)
        
        # Use training transforms for the full dataset first
        full_dataset = PlantDiseaseDataset(
            args.data_dir,
            transform=transforms_dict['train']
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(
            full_dataset,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        # Apply appropriate transforms (this is a simplification)
        train_dataset.dataset.transform = transforms_dict['train']
        val_dataset.dataset.transform = transforms_dict['val']
        test_dataset.dataset.transform = transforms_dict['val']
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if 'cuda' in args.device else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if 'cuda' in args.device else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if 'cuda' in args.device else False
        )
        
        # Get number of classes
        num_classes = len(full_dataset.classes)
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Classes: {full_dataset.classes}")
        
        # Create model
        logger.info("=" * 60)
        logger.info("Creating model...")
        logger.info("=" * 60)
        
        model = PlantDiseaseCNN(
            num_classes=num_classes,
            backbone=args.backbone,
            pretrained=True
        )
        model = model.to(args.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        trainer = CNNTrainer(
            model=model,
            device=args.device,
            output_dir=args.output,
            checkpoint_dir=config.CHECKPOINT_DIR,
            lr_scheduler=args.scheduler
        )
        
        # Train model
        logger.info("=" * 60)
        logger.info("Training...")
        logger.info("=" * 60)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_path=args.checkpoint
        )
        
        # Evaluate on test set
        logger.info("=" * 60)
        logger.info("Evaluating on test set...")
        logger.info("=" * 60)
        
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("\nTest Metrics:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
        
        # Save config
        config_dict = {
            'backbone': args.backbone,
            'num_classes': num_classes,
            'img_size': args.img_size,
            'classes': full_dataset.classes,
            'epochs_trained': args.epochs,
            'test_metrics': test_metrics
        }
        
        with open(args.output / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save training history
        with open(args.output / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {args.output}")
        logger.info(f"Artifacts:")
        logger.info(f"  - Model: {args.output / 'best_model.pt'}")
        logger.info(f"  - Config: {args.output / 'config.json'}")
        logger.info(f"  - History: {args.output / 'history.json'}")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
