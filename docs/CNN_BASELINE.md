# CNN Baseline Model

Implementation of CNN baseline models for plant disease classification with support for MobileNetV3 and EfficientNet architectures.

## Features

- **Multiple Backbones**: MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0
- **Uncertainty Estimation**: Confidence threshold-based uncertainty (returns "uncertain" if max_prob < 0.5)
- **Top-k Predictions**: Returns top-k predicted classes with confidence scores
- **Mobile Export**: ONNX and TorchScript export for on-device inference
- **Learning Rate Scheduling**: Cosine annealing, step decay, and plateau reduction schedulers
- **Model Checkpointing**: Automatic best model saving and recovery
- **Training Configuration**: Saves model config and class mappings for reproducibility

## Architecture Overview

### PlantDiseaseCNN

Main model class supporting different backbone architectures.

```python
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN

# Create model
model = PlantDiseaseCNN(
    num_classes=10,
    backbone='mobilenet_v3_small',  # or 'mobilenet_v3_large', 'efficientnet_b0'
    pretrained=True,
    uncertainty_threshold=0.5
)
```

**Characteristics**:

- Adaptive global average pooling for flexible input sizes
- Custom classifier head with batch normalization and dropout
- Pre-trained weights from ImageNet available

### PlantDiseaseDataset

Dataset loader for images organized in class subdirectories.

```python
from src.plantdisease.models.cnn_baseline import PlantDiseaseDataset

# Load dataset
dataset = PlantDiseaseDataset(
    root_dir='data/train',
    transform=train_transform
)

# Dataset structure:
# data/train/
#   ├── class1/
#   │   ├── image1.jpg
#   │   ├── image2.jpg
#   │   └── ...
#   ├── class2/
#   │   ├── image1.jpg
#   │   └── ...
```

### CNNTrainer

Comprehensive trainer class with validation, LR scheduling, and export.

```python
from src.plantdisease.models.cnn_baseline import CNNTrainer

trainer = CNNTrainer(
    model=model,
    device='cuda',
    checkpoint_dir='checkpoints',
    export_dir='exports'
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=0.001,
    scheduler_type='cosine'
)
```

## Training

### Basic Training

```bash
python scripts/03_train_cnn.py \
    --data-dir data/preprocessed_output \
    --backbone mobilenet_v3_small \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Advanced Training Options

```bash
python scripts/03_train_cnn.py \
    --data-dir data/train \
    --backbone mobilenet_v3_large \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --weight-decay 1e-4 \
    --checkpoint-dir models/checkpoints \
    --export-dir models/exports
```

### Training Script Features

- **Data Loading**: Automatic loading from class-organized directories
- **Augmentation**: Random rotation, flips, color jitter, resized crops
- **Checkpointing**: Saves best model and periodic checkpoints
- **LR Scheduling**: Cosine annealing with warmup
- **Exports**: Auto-exports ONNX and TorchScript models
- **Config Saving**: Saves class names and hyperparameters

## Evaluation

### Evaluate on Test Set

```bash
python scripts/04_evaluate_cnn.py \
    --model models/checkpoints/best_model.pt \
    --data-dir data/test \
    --batch-size 32 \
    --output metrics.json \
    --confusion-matrix confusion_matrix.png
```

### Metrics Computed

- Accuracy, Precision, Recall, F1-Score
- Per-class metrics
- Confusion matrix
- Uncertainty rate
- Classification report

## Inference

### Single Image Prediction

```bash
python scripts/05_inference_cnn.py \
    --model checkpoints/best_model.pt \
    --image test_image.jpg \
    --top-k 3 \
    --output prediction.json
```

### Batch Prediction

```bash
python scripts/05_inference_cnn.py \
    --model checkpoints/best_model.pt \
    --image-dir data/test_images \
    --top-k 3 \
    --output predictions.json
```

### Prediction Output

```json
{
  "image_path": "test_image.jpg",
  "predicted_class": "powdery_mildew",
  "predicted_class_idx": 3,
  "confidence": 0.8742,
  "is_uncertain": false,
  "top_k": [
    {
      "rank": 1,
      "class_name": "powdery_mildew",
      "class_idx": 3,
      "confidence": 0.8742
    },
    {
      "rank": 2,
      "class_name": "rust",
      "class_idx": 5,
      "confidence": 0.0923
    },
    {
      "rank": 3,
      "class_name": "healthy",
      "class_idx": 0,
      "confidence": 0.0335
    }
  ]
}
```

## API Reference

### PlantDiseaseCNN

```python
class PlantDiseaseCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'mobilenet_v3_small',
        pretrained: bool = True,
        dropout: float = 0.2,
        uncertainty_threshold: float = 0.5
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch_size, num_classes)"""

    def predict(
        self,
        logits: torch.Tensor,
        k: int = 3,
        return_uncertainty: bool = True
    ) -> Dict:
        """
        Returns:
        - class_idx: Best predicted class
        - class_prob: Confidence score
        - is_uncertain: True if class_prob < uncertainty_threshold
        - top_k_classes: Top-k class indices
        - top_k_probs: Top-k probabilities
        """

    def export_onnx(self, output_path: Path, input_shape=(1,3,224,224)):
        """Export to ONNX format for mobile deployment"""

    def export_torchscript(self, output_path: Path, method='script'):
        """Export to TorchScript format (script or trace)"""
```

### CNNTrainer

```python
class CNNTrainer:
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        scheduler_type: str = 'cosine',
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """
        Returns training history with keys:
        - train_loss, train_acc, val_loss, val_acc, learning_rate
        """

    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate on test set, returns metrics dict"""

    def export_model(
        self,
        export_formats: List[str] = ['onnx', 'torchscript'],
        model_name: str = 'plant_disease_model'
    ) -> Dict[str, Path]:
        """Export model to multiple formats"""

    def save_checkpoint(self, epoch, optimizer, loss, checkpoint_path=None):
        """Save model checkpoint"""

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint"""
```

## Model Backbones

### MobileNetV3-Small

- **Parameters**: ~2.5M
- **Inference Time**: Fast (ideal for mobile)
- **Accuracy**: Good
- **Use Case**: Real-time inference on mobile devices

### MobileNetV3-Large

- **Parameters**: ~5.4M
- **Inference Time**: Moderate
- **Accuracy**: Better than Small
- **Use Case**: Mobile with more compute

### EfficientNet-B0

- **Parameters**: ~5.3M
- **Inference Time**: Moderate
- **Accuracy**: Excellent
- **Use Case**: Balance between speed and accuracy

## Uncertainty Estimation

The model supports uncertainty estimation to identify low-confidence predictions:

```python
# Get predictions with uncertainty
pred = model.predict(logits, return_uncertainty=True)

if pred['is_uncertain']:
    # Low confidence prediction, may need review
    print(f"Uncertain: {pred['class_prob']:.2f}")
else:
    # High confidence prediction
    print(f"Confident: {pred['class_prob']:.2f}")
```

**Uncertainty Metric**: Predictions with max probability below `uncertainty_threshold` (default: 0.5) are marked as uncertain.

## Export Formats

### ONNX Export

Export for cross-platform inference (iOS, Android, Web):

```python
model.export_onnx('model.onnx')
```

**Usage**: Load with ONNX Runtime on any platform

### TorchScript Export

Export for PyTorch-based deployments:

```python
model.export_torchscript('model.pt', method='script')
```

**Methods**:

- `'script'`: Compiles Python code to TorchScript (recommended)
- `'trace'`: Traces actual execution (requires representative input)

**Usage**: Load with `torch.jit.load()` for C++ or Python inference

## Training Configuration Example

```json
{
  "model": {
    "backbone": "mobilenet_v3_small",
    "num_classes": 10,
    "uncertainty_threshold": 0.5,
    "class_names": ["healthy", "powdery_mildew", "rust", "...more classes..."]
  },
  "training": {
    "backbone": "mobilenet_v3_small",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001
  }
}
```

## Testing

Run unit tests to verify implementation:

```bash
pytest tests/test_cnn_baseline.py -v
```

**Tests Cover**:

- Model initialization with all backbones
- Forward pass and output shapes
- Uncertainty threshold functionality
- Top-k prediction functionality
- ONNX and TorchScript export
- Dataset loading
- Trainer checkpointing and loading
- Configuration saving

## Performance Tips

1. **Use larger batch sizes** (64-128) if GPU memory allows
2. **Enable mixed precision** for faster training (torch.cuda.amp)
3. **Use learning rate scheduling** (cosine annealing recommended)
4. **Normalize inputs** with ImageNet statistics
5. **Use data augmentation** during training
6. **Monitor validation loss** for early stopping

## Common Issues

### Out of Memory

Reduce batch size or use gradient accumulation:

```python
# Use smaller batch size
--batch-size 16

# Or use data parallel
model = nn.DataParallel(model)
```

### No Uncertainty Predictions

If all predictions are confident, increase uncertainty threshold:

```python
model = PlantDiseaseCNN(..., uncertainty_threshold=0.7)
```

### Export Fails

Ensure model is on CPU for export:

```python
model = model.cpu()
model.export_onnx('model.onnx')
```

## Requirements

- torch >= 1.10.0
- torchvision >= 0.11.0
- pillow >= 8.0.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

See `requirements.txt` for full dependency list.

## References

- MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02175)
- EfficientNet: [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- ONNX: [Open Neural Network Exchange](https://onnx.ai/)
- TorchScript: [PyTorch JIT Compiler](https://pytorch.org/docs/stable/jit.html)
