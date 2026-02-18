# CNN Baseline Model - Quick Reference Guide

## 🚀 Getting Started (5 minutes)

### Installation

```bash
# Install dependencies (if not already installed)
pip install torch torchvision pillow scikit-learn matplotlib seaborn
```

### Basic Training

```bash
# Train with default settings
python scripts/03_train_cnn.py --data-dir data/train

# Train with custom backbone
python scripts/03_train_cnn.py --data-dir data/train --backbone mobilenet_v3_large

# Train with more epochs
python scripts/03_train_cnn.py --data-dir data/train --epochs 100 --batch-size 16
```

### Quick Inference

```bash
# Predict on single image
python scripts/05_inference_cnn.py \
    --model checkpoints/best_model.pt \
    --image test_image.jpg

# Predict on directory
python scripts/05_inference_cnn.py \
    --model checkpoints/best_model.pt \
    --image-dir data/test_images
```

### Model Evaluation

```bash
python scripts/04_evaluate_cnn.py \
    --model checkpoints/best_model.pt \
    --data-dir data/test \
    --output metrics.json
```

---

## 📊 Model Selection Guide

### MobileNetV3-Small (Default) ✅

- **Best for**: Mobile deployment, real-time inference
- **Speed**: Fast (100+ fps)
- **Accuracy**: Good baseline
- **Use when**: You prioritize speed over accuracy

```bash
python scripts/03_train_cnn.py --backbone mobilenet_v3_small
```

### MobileNetV3-Large

- **Best for**: Better accuracy with mobile constraints
- **Speed**: Moderate (50-80 fps)
- **Accuracy**: Better than Small
- **Use when**: You need better accuracy and have more compute

```bash
python scripts/03_train_cnn.py --backbone mobilenet_v3_large
```

### EfficientNet-B0

- **Best for**: Best accuracy-speed tradeoff
- **Speed**: Moderate (40-60 fps)
- **Accuracy**: Excellent
- **Use when**: You want the best overall performance

```bash
python scripts/03_train_cnn.py --backbone efficientnet_b0
```

---

## 🔧 Common Tasks

### Train on Custom Data

```bash
# Your data structure
data/train/
├── healthy/
│   ├── image1.jpg
│   ├── image2.jpg
└── disease1/
    ├── image1.jpg
    └── image2.jpg

# Train
python scripts/03_train_cnn.py --data-dir data/train
```

### Export for Mobile

```python
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN

model = PlantDiseaseCNN(num_classes=5, backbone='mobilenet_v3_small')

# ONNX (for cross-platform)
model.export_onnx('plant_disease.onnx')  # Use on iOS, Android, Web

# TorchScript (for PyTorch)
model.export_torchscript('plant_disease.pt')  # Use with PyTorch
```

### Check Model Uncertainty

```python
model.eval()
with torch.no_grad():
    logits = model(images)
    prediction = model.predict(logits, return_uncertainty=True)

if prediction['is_uncertain']:
    print(f"Low confidence: {prediction['class_prob']:.2f}")
else:
    print(f"Confident: {prediction['class_prob']:.2f}")
```

### Fine-tune Pre-trained Model

```python
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN, CNNTrainer

# Load pre-trained model
model = PlantDiseaseCNN(
    num_classes=10,
    backbone='mobilenet_v3_small',
    pretrained=True  # Use ImageNet weights
)

trainer = CNNTrainer(model=model)
history = trainer.train(train_loader, val_loader, num_epochs=20)
```

### Get Top-3 Predictions

```python
with torch.no_grad():
    logits = model(images)
    pred = model.predict(logits, k=3)

print(f"Top 3 predictions:")
for rank, (cls_idx, prob) in enumerate(zip(pred['top_k_classes'], pred['top_k_probs']), 1):
    print(f"  {rank}. Class {cls_idx}: {prob:.4f}")
```

---

## 📈 Training Tips

### For Better Accuracy

```bash
# Larger model, longer training
python scripts/03_train_cnn.py \
    --backbone efficientnet_b0 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0001
```

### For Faster Training

```bash
# Smaller model, fewer epochs
python scripts/03_train_cnn.py \
    --backbone mobilenet_v3_small \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001
```

### Handling Imbalanced Data

- Use larger batch sizes
- Increase epochs
- Monitor validation loss
- Use class weights (future enhancement)

### Prevent Overfitting

- Use larger dropout (default: 0.2)
- Use data augmentation (enabled by default)
- Monitor validation accuracy
- Use early stopping (future enhancement)

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/03_train_cnn.py --batch-size 8

# Use smaller model
python scripts/03_train_cnn.py --backbone mobilenet_v3_small
```

### Poor Accuracy

```bash
# Increase epochs
python scripts/03_train_cnn.py --epochs 100

# Try different backbone
python scripts/03_train_cnn.py --backbone efficientnet_b0

# Reduce learning rate for fine-tuning
python scripts/03_train_cnn.py --learning-rate 0.0001
```

### Export Fails

```python
# Ensure model is on CPU
model = model.cpu()

# Try export with specific opset
model.export_onnx('model.onnx', opset_version=13)

# For TorchScript, use trace method
model.export_torchscript('model.pt', method='trace')
```

### High Uncertainty Rate

```python
# Lower the threshold
model = PlantDiseaseCNN(uncertainty_threshold=0.3)

# Or review training - might indicate poor model fit
```

---

## 🎯 Production Checklist

- [ ] Train model on full dataset
- [ ] Evaluate on test set
- [ ] Check metrics meet requirements
- [ ] Verify uncertainty rate is acceptable
- [ ] Export model to ONNX for cross-platform
- [ ] Test exported model inference
- [ ] Save training configuration
- [ ] Document class names and indices
- [ ] Test on representative images
- [ ] Monitor prediction confidence distribution
- [ ] Set up logging for inference
- [ ] Plan retraining schedule

---

## 📁 Project Structure

```
PlantDisease/
├── src/plantdisease/models/
│   ├── cnn_baseline.py           # Main implementation
│   ├── train.py                   # Training utilities
│   └── __init__.py
├── scripts/
│   ├── 03_train_cnn.py            # Training script
│   ├── 04_evaluate_cnn.py         # Evaluation script
│   ├── 05_inference_cnn.py        # Inference script
│   └── ...
├── tests/
│   ├── test_cnn_baseline.py       # Unit tests
│   └── ...
├── examples/
│   └── cnn_baseline_examples.py   # Example usage
├── docs/
│   ├── CNN_BASELINE.md            # Full documentation
│   └── ...
├── data/
│   ├── train/                     # Training images
│   ├── test/                      # Test images
│   └── ...
└── models/
    ├── checkpoints/               # Saved models
    └── exports/                   # Exported models
```

---

## 🔗 Integration with Existing Code

The CNN baseline integrates seamlessly with your project:

### Data Pipeline

```
Raw Images
    ↓ (01_preprocess_images.py)
Preprocessed Images
    ↓ (02_make_splits.py)
Train/Val/Test Splits
    ↓ (03_train_cnn.py) ← CNN Training
Models
```

### Model Comparison

Compare CNN with XGBoost baseline:

```python
from src.plantdisease.models.xgboost_baseline import XGBoostClassifier
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN

# Train both models
xgb_model = XGBoostClassifier()
cnn_model = PlantDiseaseCNN(num_classes=10)

# Compare performance
# CNN: Better for image data
# XGBoost: Good baseline, interpretable
```

### Shared Utilities

- Config management: `src.plantdisease.config`
- Logging: `src.plantdisease.utils.logger`
- Data ingestion: `src.plantdisease.data.ingestion`

---

## 📊 Expected Performance

### Baseline Metrics

- **Accuracy**: 85-95% (depends on data)
- **Precision**: 0.85-0.95
- **Recall**: 0.85-0.95
- **F1-Score**: 0.85-0.95
- **Inference Time**: 10-25ms per image (CPU)

### Factors Affecting Performance

- Data quality and quantity
- Class balance
- Image preprocessing
- Backbone selection
- Training duration
- Hyperparameter tuning

---

## 🎓 Learning Resources

### In This Package

- Full examples: `examples/cnn_baseline_examples.py`
- Unit tests: `tests/test_cnn_baseline.py`
- Documentation: `docs/CNN_BASELINE.md`
- Training code: `scripts/03_train_cnn.py`

### External Resources

- PyTorch Documentation: https://pytorch.org/docs
- MobileNetV3: https://arxiv.org/abs/1905.02175
- EfficientNet: https://arxiv.org/abs/1905.11946

---

## 🆘 Support

### Getting Help

1. Check `docs/CNN_BASELINE.md` for detailed documentation
2. Review `examples/cnn_baseline_examples.py` for working code
3. Run tests: `pytest tests/test_cnn_baseline.py -v`
4. Check script help: `python scripts/03_train_cnn.py --help`

### Common Commands Reference

```bash
# Training
python scripts/03_train_cnn.py --help

# Evaluation
python scripts/04_evaluate_cnn.py --help

# Inference
python scripts/05_inference_cnn.py --help

# Testing
pytest tests/test_cnn_baseline.py -v
```

---

## 📝 Notes

- Models are trained on ImageNet-normalized images
- Default input size: 224×224
- Pre-trained weights available from torchvision
- ONNX export uses opset 13 (good compatibility)
- TorchScript supports both script and trace methods
- Configuration files saved automatically with models

---

**Last Updated**: 2026-02-18
**Version**: 1.0
**Status**: Production Ready
