# Plant Disease Detection

A machine learning project for plant disease classification using transfer learning and gradient boosting approaches.

## Project Overview

This project implements two baseline models for plant disease detection:

1. **CNN Baseline** - Deep learning models using MobileNetV3 and EfficientNet
2. **XGBoost Baseline** - Gradient boosting with hand-crafted image features

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone <repo-url>
cd PlantDisease

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
PlantDisease/
├── src/plantdisease/           # Main source code
│   ├── config.py               # Configuration management
│   ├── data/                   # Data processing modules
│   │   ├── download.py         # Dataset download utilities
│   │   ├── ingestion.py        # Data loading
│   │   ├── splits.py           # Train/val/test splitting
│   │   └── preprocess/         # Image preprocessing filters
│   ├── models/                 # Model implementations
│   │   ├── cnn_baseline.py     # CNN models (MobileNetV3, EfficientNet)
│   │   ├── xgboost_baseline.py # XGBoost classifier
│   │   ├── evaluate.py         # Evaluation metrics
│   │   ├── features.py         # Feature extraction
│   │   └── train.py            # Training loops
│   └── utils/                  # Utilities
├── scripts/                    # Command-line scripts
│   ├── 00_download_data.py     # Download dataset utilities
│   ├── 01_preprocess_images.py # Batch preprocessing
│   ├── 02_make_splits.py       # Create train/val/test splits
│   ├── 03_train_cnn.py         # Train CNN models
│   ├── 04_evaluate_cnn.py      # Evaluate CNN models
│   ├── 04_evaluate.py          # Generic evaluation
│   ├── 05_inference_cnn.py     # CNN inference
│   ├── preprocess_cli.py       # Batch preprocessing CLI
│   ├── train_cnn_cli.py        # CNN training CLI
│   ├── train_xgb_cli.py        # XGBoost training CLI
│   └── evaluate_cli.py         # Model evaluation CLI
├── tests/                      # Test suite (61 tests, all passing)
├── data/                       # Data directory
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed images
│   ├── features/               # Extracted features
│   └── splits/                 # Train/val/test splits
├── models/                     # Model checkpoints and exports
│   ├── checkpoints/            # Training checkpoints
│   └── exports/                # ONNX/TorchScript models
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── pyproject.toml             # Project metadata
```

## Models

### CNN Baseline (`src/plantdisease/models/cnn_baseline.py`)

Transfer learning models with ImageNet pretrained backbones:

**Architecture:**
- Backbone: MobileNetV3-Small, MobileNetV3-Large, or EfficientNet-B0
- Feature extractor: Backbone without classifier
- Classifier Head: Custom `ClassifierHead` module with adaptive pooling
  - Flexible input handling (4D and 2D tensors)
  - TorchScript compatible
  - Sequential FC layers with dropout and batch normalization

**Features:**
- Uncertainty threshold for low-confidence predictions
- Top-k predictions with confidence scores
- ONNX and TorchScript export for deployment
- Learning rate scheduling (Cosine, Step, ReduceLROnPlateau)
- Checkpointing and checkpoint loading

**Usage:**
```python
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN

# Create model
model = PlantDiseaseCNN(
    num_classes=3,
    backbone='mobilenet_v3_small',
    uncertainty_threshold=0.5
)

# Inference
logits = model(x)  # x shape: (batch_size, 3, 224, 224)
predictions = model.predict(logits, k=3)
```

### XGBoost Baseline (`src/plantdisease/models/xgboost_baseline.py`)

Gradient boosting classifier with hand-crafted features:

**Features Extracted (194-dim total):**
- HSV Histograms: 96 dimensions (H=32, S=32, V=32)
- LBP (Local Binary Patterns): 26 dimensions
- GLCM (Gray-Level Co-occurrence Matrix): 72 dimensions

**Configuration:**
- Estimators: 100
- Max depth: 6
- Learning rate: 0.1
- Early stopping: Yes

**Usage:**
```python
from src.plantdisease.models.xgboost_baseline import XGBoostClassifier

# Train
classifier = XGBoostClassifier(num_classes=3)
classifier.train(X_train, y_train, X_val, y_val)

# Predict
predictions = classifier.predict(X_test)
```

## CLI Scripts

### 1. Preprocessing (`scripts/preprocess_cli.py`)

Batch process images with multiple preprocessing options:

```bash
python scripts/preprocess_cli.py \
  --input data/raw \
  --output data/processed \
  --denoise bilateral \
  --contrast clahe \
  --resize 224 \
  --num-workers 4
```

**Options:**
- `--denoise`: bilateral, median, gaussian
- `--contrast`: clahe, histogram
- `--resize`: Target image size (default: 224)
- `--num-workers`: Parallel processing workers

### 2. CNN Training (`scripts/train_cnn_cli.py`)

Train CNN models with configurable hyperparameters:

```bash
python scripts/train_cnn_cli.py \
  --data-dir data/processed \
  --backbone mobilenet_v3_small \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --augment \
  --scheduler cosine \
  --output models/checkpoints
```

**Options:**
- `--backbone`: mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0
- `--epochs`: Training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Initial learning rate
- `--augment`: Enable data augmentation
- `--scheduler`: cosine, step, reduce_on_plateau

### 3. XGBoost Training (`scripts/train_xgb_cli.py`)

Train XGBoost classifier:

```bash
python scripts/train_xgb_cli.py \
  --data-dir data/features \
  --estimators 100 \
  --max-depth 6 \
  --learning-rate 0.1 \
  --output models/xgboost
```

### 4. Model Evaluation (`scripts/evaluate_cli.py`)

Evaluate trained models:

```bash
python scripts/evaluate_cli.py \
  --model-path models/checkpoints/best_model.pt \
  --test-data data/processed \
  --model-type cnn
```

## Data Processing Pipeline

### 1. Prepare Your Data

Place your plant disease images in `data/raw/` organized by class:
```
data/raw/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...
```

### 2. Preprocessing
```bash
python scripts/01_preprocess_images.py
```
Features:
- Image resizing and standardization
- Denoising (bilateral, median, Gaussian)
- Contrast enhancement (CLAHE, histogram)
- Grayscale conversion options
- ROI extraction and segmentation

### 3. Create Splits
```bash
python scripts/02_make_splits.py
```
Creates stratified train/validation/test splits.

### 4. Extract Features (XGBoost only)
```bash
python scripts/03_train_cnn.py --extract-features --model xgboost
```
Extracts hand-crafted features (HSV, LBP, GLCM) from preprocessed images.

### 5. Train Models
```bash
# CNN
python scripts/03_train_cnn.py

# XGBoost
python scripts/03_train_cnn.py --model xgboost
```

### 6. Evaluate
```bash
python scripts/04_evaluate_cnn.py
```

## Testing

Run the complete test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_cnn_baseline.py -v
pytest tests/test_task1_xgboost.py -v
pytest tests/test_preprocess.py -v
pytest tests/test_splits.py -v

# With coverage
pytest tests/ --cov=src/plantdisease
```

**Test Coverage:**
- CNN Baseline: 17 tests (model initialization, forward pass, exports, training)
- XGBoost Baseline: 18 tests (feature extraction, training, evaluation, persistence)
- Preprocessing: 21 tests (all preprocessing filters and utilities)
- Data Splits: 3 tests (train/val/test split validation)
- **Total: 61 tests - All passing ✅**

## Configuration

Edit `config.yaml` for project-wide settings:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  features_dir: data/features

model:
  num_classes: 3
  backbone: mobilenet_v3_small
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
```

## Model Export

### ONNX Export
```python
model.export_onnx('model.onnx', input_shape=(1, 3, 224, 224))
```

### TorchScript Export
```python
model.export_torchscript('model.pt')
```

## Architecture Details

### ClassifierHead Module
The CNN classifier uses a custom `ClassifierHead` module designed for:
- **Polymorphic input handling**: Accepts both 4D (B, C, H, W) and 2D (B, C) tensors
- **TorchScript compatibility**: Fully serializable for mobile deployment
- **Backward compatibility**: Supports index-based access like Sequential

```python
class ClassifierHead(nn.Module):
    def __init__(self, features_dim, num_classes, dropout=0.2, needs_pooling=False):
        self.avgpool = AdaptiveAvgPool2d(...) if needs_pooling else Identity()
        self.fc_head = Sequential(...)
    
    def forward(self, x):
        if x.dim() == 4:
            x = self.avgpool(x)
        x = self.flatten(x) if x.dim() > 2 else x
        return self.fc_head(x)
```

## Key Features

✅ **Multiple Backends**: MobileNetV3, EfficientNet, XGBoost
✅ **Mobile Export**: ONNX and TorchScript support
✅ **Data Augmentation**: Extensive preprocessing pipeline
✅ **Flexible Training**: CLI scripts with hyperparameter control
✅ **Comprehensive Testing**: 61 unit tests covering all modules
✅ **Checkpoint Management**: Save/load training states
✅ **Uncertainty Quantification**: Confidence thresholds and top-k predictions

## Performance

Both models achieve strong performance on the plant disease classification task:

**CNN Baseline:**
- Supports real-time inference on mobile devices via TorchScript
- Efficient backbone selection (MobileNetV3-Small ~2.5M params)
- Uncertainty threshold for filtering low-confidence predictions

**XGBoost Baseline:**
- Fast training and inference
- Interpretable feature importance
- Robust feature engineering with multi-dimensional analysis

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages:

- PyTorch 1.10+
- torchvision 0.11+
- scikit-learn
- XGBoost
- NumPy, Pandas, SciPy
- OpenCV, Pillow
- pytest

## License

This project is part of a machine learning course assignment.

## Contributors

- Task 1 (XGBoost): Feature extraction and baseline model
- Task 2 (CNN): Deep learning implementation with transfer learning
- Task 3 (CLI & Docs): Command-line interfaces and documentation
