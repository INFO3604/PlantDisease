"""
Example: Complete workflow for CNN baseline model.

This script demonstrates:
1. Creating datasets from image directories
2. Training a CNN model with different backbones
3. Making predictions with uncertainty estimation
4. Exporting models for mobile deployment
5. Evaluating model performance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from pathlib import Path

from src.plantdisease.models.cnn_baseline import (
    PlantDiseaseCNN,
    PlantDiseaseDataset,
    CNNTrainer
)
from src.plantdisease import config


def example_1_basic_inference():
    """Example 1: Create model and make predictions."""
    print("\n" + "="*60)
    print("Example 1: Basic Inference")
    print("="*60)
    
    # Create a simple model
    model = PlantDiseaseCNN(
        num_classes=5,
        backbone='mobilenet_v3_small',
        pretrained=False
    )
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        predictions = model.predict(logits, k=3)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"\nPredictions for first sample:")
    print(f"  Predicted class: {predictions['class_idx']}")
    print(f"  Confidence: {predictions['class_prob']:.4f}")
    print(f"  Is uncertain: {predictions['is_uncertain']}")
    print(f"  Top-3 classes: {predictions['top_k_classes']}")
    print(f"  Top-3 probs: {predictions['top_k_probs']}")


def example_2_different_backbones():
    """Example 2: Compare different backbone architectures."""
    print("\n" + "="*60)
    print("Example 2: Different Backbones")
    print("="*60)
    
    backbones = [
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'efficientnet_b0'
    ]
    
    x = torch.randn(1, 3, 224, 224)
    
    for backbone in backbones:
        model = PlantDiseaseCNN(
            num_classes=10,
            backbone=backbone,
            pretrained=False
        )
        model.eval()
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # Inference time
        with torch.no_grad():
            logits = model(x)
        
        print(f"\n{backbone}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {logits.shape}")


def example_3_uncertainty():
    """Example 3: Using uncertainty threshold."""
    print("\n" + "="*60)
    print("Example 3: Uncertainty Estimation")
    print("="*60)
    
    model = PlantDiseaseCNN(
        num_classes=3,
        backbone='mobilenet_v3_small',
        pretrained=False,
        uncertainty_threshold=0.5
    )
    model.eval()
    
    # Create logits for different confidence levels
    # High confidence: [0.8, 0.15, 0.05]
    logits_high = torch.tensor([[2.0, 0.3, -0.5]], dtype=torch.float32)
    
    # Low confidence: [0.35, 0.35, 0.3]
    logits_low = torch.tensor([[0.2, 0.2, 0.1]], dtype=torch.float32)
    
    with torch.no_grad():
        pred_high = model.predict(logits_high, return_uncertainty=True)
        pred_low = model.predict(logits_low, return_uncertainty=True)
    
    print(f"\nHigh confidence prediction:")
    print(f"  Confidence: {pred_high['class_prob']:.4f}")
    print(f"  Is uncertain: {pred_high['is_uncertain']}")
    
    print(f"\nLow confidence prediction:")
    print(f"  Confidence: {pred_low['class_prob']:.4f}")
    print(f"  Is uncertain: {pred_low['is_uncertain']}")


def example_4_export_models():
    """Example 4: Export models for deployment."""
    print("\n" + "="*60)
    print("Example 4: Model Export")
    print("="*60)
    
    model = PlantDiseaseCNN(
        num_classes=5,
        backbone='mobilenet_v3_small',
        pretrained=False
    )
    
    # Create temporary export directory
    export_dir = Path('temp_exports')
    export_dir.mkdir(exist_ok=True)
    
    try:
        # ONNX export
        print("\nExporting to ONNX...")
        onnx_path = export_dir / 'model.onnx'
        model.export_onnx(onnx_path)
        print(f"  ✓ Saved to {onnx_path}")
        print(f"  Size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # TorchScript export
        print("\nExporting to TorchScript...")
        ts_path = export_dir / 'model.pt'
        model.export_torchscript(ts_path, method='trace')
        print(f"  ✓ Saved to {ts_path}")
        print(f"  Size: {ts_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Verify TorchScript model
        print("\nVerifying TorchScript model...")
        loaded_model = torch.jit.load(ts_path)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = loaded_model(x)
        print(f"  ✓ TorchScript inference successful, output shape: {output.shape}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(export_dir, ignore_errors=True)


def example_5_training_setup():
    """Example 5: Setup training (without actual training)."""
    print("\n" + "="*60)
    print("Example 5: Training Setup")
    print("="*60)
    
    # Create model
    model = PlantDiseaseCNN(
        num_classes=10,
        backbone='mobilenet_v3_small',
        pretrained=True
    )
    
    # Create trainer
    trainer = CNNTrainer(
        model=model,
        device='cpu',
        checkpoint_dir='./checkpoints',
        export_dir='./exports'
    )
    
    print(f"\nTrainer created successfully!")
    print(f"  Device: {trainer.device}")
    print(f"  Checkpoint dir: {trainer.checkpoint_dir}")
    print(f"  Export dir: {trainer.export_dir}")
    
    # Show training configuration
    print(f"\nTraining can be started with:")
    print(f"  history = trainer.train(")
    print(f"      train_loader=train_loader,")
    print(f"      val_loader=val_loader,")
    print(f"      num_epochs=50,")
    print(f"      learning_rate=0.001,")
    print(f"      scheduler_type='cosine'")
    print(f"  )")


def example_6_class_names():
    """Example 6: Working with class names."""
    print("\n" + "="*60)
    print("Example 6: Class Names and Labels")
    print("="*60)
    
    # Example class names
    class_names = [
        'healthy',
        'powdery_mildew',
        'rust',
        'leaf_spot',
        'blight'
    ]
    
    # Create model
    model = PlantDiseaseCNN(
        num_classes=len(class_names),
        backbone='mobilenet_v3_small',
        pretrained=False
    )
    model.eval()
    
    # Create random logits
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        pred = model.predict(logits, k=3)
    
    print(f"\nClass names: {class_names}")
    print(f"\nPrediction results:")
    print(f"  Predicted: {class_names[pred['class_idx']]}")
    print(f"  Confidence: {pred['class_prob']:.4f}")
    
    print(f"\n  Top-3 predictions:")
    for rank, (class_idx, prob) in enumerate(
        zip(pred['top_k_classes'], pred['top_k_probs']), 1
    ):
        print(f"    {rank}. {class_names[class_idx]}: {prob:.4f}")


def example_7_top_k_variations():
    """Example 7: Top-k predictions with different k values."""
    print("\n" + "="*60)
    print("Example 7: Top-k Predictions")
    print("="*60)
    
    model = PlantDiseaseCNN(
        num_classes=10,
        backbone='mobilenet_v3_small',
        pretrained=False
    )
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(x)
    
    for k in [1, 3, 5]:
        with torch.no_grad():
            pred = model.predict(logits, k=k)
        
        print(f"\nTop-{k} predictions:")
        print(f"  Classes: {pred['top_k_classes']}")
        print(f"  Probs: {pred['top_k_probs']}")
        print(f"  Sum of probs: {pred['top_k_probs'].sum():.6f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CNN BASELINE MODEL EXAMPLES")
    print("="*60)
    
    # Run examples
    example_1_basic_inference()
    example_2_different_backbones()
    example_3_uncertainty()
    example_4_export_models()
    example_5_training_setup()
    example_6_class_names()
    example_7_top_k_variations()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
