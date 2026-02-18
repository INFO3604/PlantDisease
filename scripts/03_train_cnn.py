#!/usr/bin/env python
"""
Training script for CNN baseline model with MobileNetV3 and EfficientNet.

Usage:
    python scripts/03_train_cnn.py --backbone mobilenet_v3_small --epochs 50 --batch-size 32
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.models.cnn_baseline import (
    PlantDiseaseCNN, PlantDiseaseDataset, CNNTrainer
)
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)


def create_transforms(img_size: int = 224) -> dict:
    """Create image transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return {'train': train_transform, 'val': val_transform}


def load_data(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1
) -> tuple:
    """Load training and validation data."""
    
    transform = create_transforms()
    
    # Load dataset
    dataset = PlantDiseaseDataset(
        root_dir=data_dir,
        transform=transform['train']
    )
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = transform['val']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, dataset.get_class_names()


def train_model(
    data_dir: Path,
    backbone: str = 'mobilenet_v3_small',
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    checkpoint_dir: Optional[Path] = None,
    export_dir: Optional[Path] = None
) -> None:
    """Train CNN model."""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    export_dir = export_dir or config.EXPORT_DIR
    
    # Load data
    logger.info(f"Loading data from {data_dir}")
    train_loader, val_loader, class_names = load_data(
        data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Create model
    logger.info(f"Creating model with backbone: {backbone}")
    model = PlantDiseaseCNN(
        num_classes=len(class_names),
        backbone=backbone,
        pretrained=True,
        dropout=0.2,
        uncertainty_threshold=0.5
    )
    
    # Create trainer
    trainer = CNNTrainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        export_dir=export_dir
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_type='cosine'
    )
    
    # Save training config
    trainer.save_training_config(
        config_path=checkpoint_dir / 'training_config.json',
        class_names=class_names,
        config_dict={
            'backbone': backbone,
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
    )
    
    # Export model
    logger.info("Exporting model...")
    export_paths = trainer.export_model(
        export_formats=['onnx', 'torchscript'],
        model_name=f'plant_disease_{backbone}'
    )
    
    logger.info("Training completed!")
    logger.info(f"Model exported to:")
    for fmt, path in export_paths.items():
        logger.info(f"  {fmt.upper()}: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train CNN baseline for plant disease detection'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/preprocessed_output'),
        help='Directory containing training images organized by class'
    )
    parser.add_argument(
        '--backbone',
        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0'],
        default='mobilenet_v3_small',
        help='Backbone architecture'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='L2 regularization'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--export-dir',
        type=Path,
        default=None,
        help='Directory to save exported models'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        backbone=args.backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        export_dir=args.export_dir
    )


if __name__ == "__main__":
    main()
