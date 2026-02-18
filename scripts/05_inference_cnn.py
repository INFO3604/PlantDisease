"""
Inference script for trained CNN model.

Usage:
    python scripts/04_inference_cnn.py --model checkpoints/best_model.pt --image test_image.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN
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
    logger.info(f"  Backbone: {model_config.get('backbone')}")
    logger.info(f"  Classes: {model_config.get('num_classes')}")
    logger.info(f"  Uncertainty threshold: {model_config.get('uncertainty_threshold')}")
    
    return model, model_config


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path) as f:
        return json.load(f)


def preprocess_image(image_path: Path, img_size: int = 224) -> torch.Tensor:
    """Load and preprocess image."""
    
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_single_image(
    model: PlantDiseaseCNN,
    image_path: Path,
    class_names: Optional[list] = None,
    device: str = 'cpu',
    k: int = 3
) -> Dict[str, Any]:
    """Make prediction on single image."""
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(image_tensor)
    
    # Get predictions
    prediction = model.predict(logits, k=k, return_uncertainty=True)
    
    # Format output
    result = {
        'image_path': str(image_path),
        'predicted_class_idx': int(prediction['class_idx']),
        'confidence': float(prediction['class_prob']),
        'is_uncertain': bool(prediction['is_uncertain']),
        'top_k': []
    }
    
    if class_names:
        result['predicted_class'] = class_names[prediction['class_idx']]
    
    # Add top-k predictions
    for idx, (class_idx, prob) in enumerate(
        zip(prediction['top_k_classes'], prediction['top_k_probs'])
    ):
        top_k_item = {
            'rank': idx + 1,
            'class_idx': int(class_idx),
            'confidence': float(prob)
        }
        
        if class_names:
            top_k_item['class_name'] = class_names[class_idx]
        
        result['top_k'].append(top_k_item)
    
    return result


def batch_predict(
    model: PlantDiseaseCNN,
    image_dir: Path,
    class_names: Optional[list] = None,
    device: str = 'cpu',
    extensions: tuple = ('.jpg', '.jpeg', '.png')
) -> list:
    """Make predictions on all images in a directory."""
    
    results = []
    image_files = []
    
    # Collect image files
    for ext in extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
    
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return results
    
    logger.info(f"Found {len(image_files)} images for prediction")
    
    for image_path in image_files:
        try:
            result = predict_single_image(
                model=model,
                image_path=image_path,
                class_names=class_names,
                device=device
            )
            results.append(result)
            
            logger.info(
                f"  {image_path.name}: {result['predicted_class_idx']} "
                f"(confidence: {result['confidence']:.4f})"
            )
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Inference with trained CNN model')
    parser.add_argument('--model', type=Path, required=True, help='Path to checkpoint')
    parser.add_argument('--image', type=Path, help='Path to single image')
    parser.add_argument('--image-dir', type=Path, help='Directory with images')
    parser.add_argument('--config', type=Path, help='Path to training config')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu', help='Device')
    parser.add_argument('--top-k', type=int, default=3, help='Return top-k predictions')
    parser.add_argument('--output', type=Path, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        logger.error("Provide either --image or --image-dir")
        sys.exit(1)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model, model_config = load_model(args.model, device=args.device)
    
    # Load class names
    class_names = None
    if args.config and args.config.exists():
        training_config = load_config(args.config)
        class_names = training_config.get('model', {}).get('class_names')
    
    # Make predictions
    if args.image:
        logger.info(f"Predicting on single image: {args.image}")
        result = predict_single_image(
            model=model,
            image_path=args.image,
            class_names=class_names,
            device=args.device,
            k=args.top_k
        )
        
        logger.info("\nPrediction Results:")
        logger.info(f"  Predicted class: {result.get('predicted_class', result['predicted_class_idx'])}")
        logger.info(f"  Confidence: {result['confidence']:.4f}")
        logger.info(f"  Uncertain: {'Yes' if result['is_uncertain'] else 'No'}")
        logger.info(f"\n  Top-{args.top_k} Predictions:")
        for item in result['top_k']:
            class_str = f"{item.get('class_name', item['class_idx'])}"
            logger.info(f"    {item['rank']}. {class_str}: {item['confidence']:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nResults saved to {args.output}")
    
    elif args.image_dir:
        logger.info(f"Predicting on directory: {args.image_dir}")
        results = batch_predict(
            model=model,
            image_dir=args.image_dir,
            class_names=class_names,
            device=args.device
        )
        
        # Summary statistics
        if results:
            uncertain_count = sum(1 for r in results if r['is_uncertain'])
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            logger.info(f"\nSummary Statistics:")
            logger.info(f"  Total images: {len(results)}")
            logger.info(f"  Uncertain predictions: {uncertain_count} ({100*uncertain_count/len(results):.1f}%)")
            logger.info(f"  Average confidence: {avg_confidence:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
