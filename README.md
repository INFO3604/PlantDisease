# Plant Disease Detection

A machine learning project for automated plant disease detection and classification on Solanaceae crops (pepper, potato, tomato) using the PlantVillage dataset. Combines a custom image preprocessing pipeline with transfer learning (CNN) and six traditional ML classifiers.

## Project Overview

This project implements:

1. **Image Preprocessing Pipeline** - Automated leaf segmentation, disease detection & severity quantification
2. **CNN (Transfer Learning)** - MobileNetV3-Small with ImageNet pretrained weights (**99.93% accuracy**)
3. **Traditional ML Classifiers (6 models)** - XGBoost, CatBoost, Random Forest, SVM, Logistic Regression, KNN using 109 hand-crafted features (Gabor texture, CIELAB + HSV colour, morphology)
4. **Feature Extraction** - 109-dimensional feature vector from preprocessed leaf images

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
├── src/plantdisease/               # Main source code
│   ├── config.py                   # Configuration management
│   ├── data/                       # Data processing modules
│   │   ├── ingestion.py            # Data loading
│   │   ├── splits.py               # Train/val/test splitting
│   │   └── preprocess/             # Image preprocessing
│   │       ├── pipeline.py         # PreprocessingPipeline (rembg → shadow → disease → normalise)
│   │       ├── background.py       # rembg deep-learning background removal (session caching)
│   │       ├── shadow.py           # HSV shadow detection and removal
│   │       ├── disease.py          # HSV disease segmentation + severity
│   │       ├── denoise.py          # Denoising utilities
│   │       ├── grayscale.py        # Grayscale conversion utilities
│   │       └── resize_standardize.py # Resize and standardization
│   ├── features/                   # Feature extraction
│   │   └── extract_features.py     # 109-feature extractor (Gabor, CIELAB+HSV, morphology)
│   ├── models/                     # Model implementations
│   │   ├── cnn_baseline.py         # CNN models (MobileNetV3, EfficientNet)
│   │   ├── train_rf.py             # Random Forest classifier
│   │   ├── train_svm.py            # SVM (RBF kernel) classifier
│   │   ├── train_logistic_regression.py  # Logistic Regression classifier
│   │   ├── train_knn.py            # K-Nearest Neighbors classifier
│   │   ├── train_xgboost.py        # XGBoost classifier
│   │   ├── train_catboost.py       # CatBoost classifier
│   │   ├── rf_ensemble.py          # Random Forest ensemble (backward-compat shim)
│   │   ├── evaluate.py             # Evaluation metrics
│   │   ├── train.py                # Training loops
│   │   └── utils.py                # Model utilities
│   └── utils/                      # Utilities (logging, paths)
├── scripts/                        # Command-line scripts
│   ├── demo_single_image.py        # Demo: process images & output 3×3 visualization grids
│   ├── extract_features_dataset.py # Extract 109 features from full dataset → features.csv
│   ├── evaluate_all_classifiers.py # Evaluate all 6 classifiers on features.csv
│   ├── train_cnn_cli.py            # CNN training CLI
│   ├── train_rf_cli.py             # Random Forest training CLI
│   ├── evaluate_cli.py             # Model evaluation CLI
│   ├── 02_make_splits.py           # Create train/val/test splits
│   ├── 03_train_cnn.py             # Train CNN models
│   ├── 04_evaluate_cnn.py          # Evaluate CNN models
│   └── 05_inference_cnn.py         # CNN inference
├── tests/                          # Test suite
│   ├── test_classifiers.py         # All 6 traditional classifier tests
│   ├── test_cnn_baseline.py        # CNN tests
│   ├── test_preprocess.py          # Preprocessing tests
│   ├── test_rf_ensemble.py         # Random Forest ensemble tests
│   └── test_splits.py              # Data split tests
├── models/                         # Trained model artifacts
│   ├── checkpoints/                # Training checkpoints
│   │   ├── best_model.pt           # Best CNN checkpoint
│   │   ├── checkpoint_epoch_10.pt  # Epoch 10 checkpoint
│   │   └── training_config.json    # Training configuration
│   └── exports/                    # Exported models for deployment
│       ├── plant_disease_mobilenet_v3_small.onnx  # ONNX format
│       └── plant_disease_mobilenet_v3_small.pt    # TorchScript format
├── data/                           # Data directory
│   ├── raw/
│   │   └── train/                  # PlantVillage dataset (10 classes, ~18,648 images)
│   │       ├── Pepper,_bell___Bacterial_spot/
│   │       ├── Pepper,_bell___healthy/
│   │       ├── Potato___Early_blight/
│   │       ├── Potato___healthy/
│   │       ├── Potato___Late_blight/
│   │       ├── Tomato___Bacterial_spot/
│   │       ├── Tomato___Early_blight/
│   │       ├── Tomato___Late_blight/
│   │       ├── Tomato___Septoria_leaf_spot/
│   │       └── Tomato___Target_Spot/
│   ├── processed/
│   │   └── features.csv            # Extracted features (5000 samples, 109 features)
│   ├── demo_input/                 # Drop images here for manual demo processing
│   ├── demo_output/                # Demo visualization output
│   └── features/                   # Extracted features (.npz)
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
└── PREPROCESSING_README.md         # Detailed preprocessing pipeline docs
```

## Image Preprocessing Pipeline

The core preprocessing pipeline (`src/plantdisease/data/preprocess/pipeline.py`) provides automated background removal, disease detection, and severity quantification. It uses **rembg / U2-Net** deep-learning background removal, **HSV shadow removal**, and **HSV disease segmentation** (yellow chlorosis + brown necrosis detection).

### Pipeline Stages

| Step | Method | Purpose |
|------|--------|---------|
| 1. Remove Background | rembg / U2-Net (deep learning) | Background removal → RGBA with leaf mask from alpha |
| 2. Resize | Lanczos-4 (300×300) | Spatial standardisation preserving alpha channel |
| 3. Shadow Removal | HSV thresholds (V<80, S<50) | Shadow detection and correction on leaf surface |
| 4. Disease Segmentation | HSV colour ranges (yellow, brown, dark necrotic) | Diseased region detection with severity metrics |
| 5. Severity | Diseased pixels / total leaf pixels | Percentage-based severity quantification |
| 6. Normalisation | Pixel values → [0, 1] float32 | Standardised input for downstream models |

### Quick Demo

```bash
# Auto-detect mode: processes data/demo_input/ if it has images,
# otherwise samples 20 random images from the dataset:
python scripts/demo_single_image.py

# Manual mode: place your own images in data/demo_input/ and run:
python scripts/demo_single_image.py --input data/demo_input

# Dataset mode with custom sample count:
python scripts/demo_single_image.py -n 10

# Reproducible run:
python scripts/demo_single_image.py -n 20 --seed 42
```

**Manual mode**: Drop images into `data/demo_input/` — all images are processed.
**Dataset mode**: When `demo_input/` is empty, samples N images (stratified across classes) from the PlantVillage dataset.

Both modes save a 3×3 visualization grid per image and export 109 features + class labels to `features.csv`.

### Python API

```python
import cv2
from plantdisease.data.preprocess import PreprocessingPipeline

pipe = PreprocessingPipeline()
image = cv2.imread("path/to/leaf.jpg")
result = pipe.run(image)

print(f"Severity: {result.severity_percent:.1f}%")
print(f"Yellow pixels: {result.yellow_pixels}")
print(f"Brown pixels: {result.brown_pixels}")
cv2.imwrite("overlay.jpg", result.disease_overlay)
```

See [PREPROCESSING_README.md](PREPROCESSING_README.md) for full documentation including HSV threshold details, tuning guide, and design decisions.

---

## Model Performance

### Accuracy Comparison

| Model | Type | Accuracy |
|-------|------|----------|
| **CNN (MobileNetV3-Small)** | Deep Learning | **99.93%** |
| SVM (RBF kernel) | Traditional ML | 88.50% |
| XGBoost | Traditional ML | 88.10% |
| CatBoost | Traditional ML | 87.50% |
| Logistic Regression | Traditional ML | 87.10% |
| KNN (k=6, RF-weighted) | Traditional ML | 85.50% |
| Random Forest | Traditional ML | 85.10% |

- **CNN**: Trained on 18,648 images (85/15 train/val split), 10 epochs, MobileNetV3-Small backbone with ImageNet pretrained weights.
- **Traditional classifiers**: Evaluated on 5,000 sampled images using 109 hand-crafted features with 80/20 stratified train/test split. All 6 classifiers achieve ≥85% accuracy.

---

## Models

### CNN (`src/plantdisease/models/cnn_baseline.py`)

Transfer learning model with ImageNet pretrained backbone — achieves **99.93% validation accuracy**.

**Architecture:**
- Backbone: MobileNetV3-Small (pretrained on ImageNet)
- Input: 224×224 RGB images
- Classifier Head: Custom `ClassifierHead` with adaptive pooling, batch normalization, dropout (0.2)
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: Cosine annealing
- 10 classes, ~2.5M parameters

**Features:**
- Uncertainty threshold for low-confidence predictions
- Top-k predictions with confidence scores
- ONNX and TorchScript export for deployment
- Learning rate scheduling (Cosine, Step, ReduceLROnPlateau)
- Checkpointing and checkpoint loading

**Training:**
```bash
python scripts/03_train_cnn.py \
  --backbone mobilenet_v3_small \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --data-dir data/raw/train
```

**Usage:**
```python
from src.plantdisease.models.cnn_baseline import PlantDiseaseCNN

# Create model
model = PlantDiseaseCNN(
    num_classes=10,
    backbone='mobilenet_v3_small',
    uncertainty_threshold=0.5
)

# Inference
logits = model(x)  # x shape: (batch_size, 3, 224, 224)
predictions = model.predict(logits, k=3)
```

### Traditional ML Classifiers

Six classifiers using the 109-dimensional hand-crafted feature vector:

| Classifier | Module | Key Parameters |
|------------|--------|------------------|
| Random Forest | `train_rf.py` | 1000 trees, no depth limit, sqrt features |
| SVM | `train_svm.py` | RBF kernel, C=10, StandardScaler pipeline |
| Logistic Regression | `train_logistic_regression.py` | C=10, max_iter=5000, StandardScaler pipeline |
| KNN | `train_knn.py` | k=6, RF-importance weighting, manhattan distance |
| XGBoost | `train_xgboost.py` | 600 estimators, max_depth=10, lr=0.03, GPU |
| CatBoost | `train_catboost.py` | 800 iterations, depth=10, lr=0.03, GPU |

**Feature Vector (109 dimensions):**
- Disease Gabor texture: 36 dimensions (12 filter banks × 3 statistics: mean, std, energy)
- Leaf Gabor texture: 36 dimensions (whole-leaf Gabor for healthy plant discrimination)
- Disease CIELAB + HSV colour: 12 dimensions (L*, a*, b*, H, S, V mean + std)
- Leaf CIELAB + HSV colour: 12 dimensions (whole-leaf colour stats)
- Severity ratios: 3 dimensions (disease, yellow, brown)
- Morphology: 10 dimensions (lesion count, area, perimeter, shape descriptors)

**Feature Extraction:**
```bash
# Extract features from full dataset → data/processed/features.csv
python scripts/extract_features_dataset.py
```

## CLI Scripts

### 1. CNN Training (`scripts/03_train_cnn.py`)

Train CNN models with configurable hyperparameters:

```bash
python scripts/03_train_cnn.py \
  --data-dir data/raw/train \
  --backbone mobilenet_v3_small \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001
```

**Options:**
- `--backbone`: mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--data-dir`: Path to the training data directory

### 2. Feature Extraction (`scripts/extract_features_dataset.py`)

Extract 109 hand-crafted features from the dataset for traditional ML classifiers:

```bash
python scripts/extract_features_dataset.py
```

Outputs `data/processed/features.csv` with 109 features + image_id + label columns.

### 3. Random Forest Training (`scripts/train_rf_cli.py`)

Train Random Forest ensemble:

```bash
python scripts/train_rf_cli.py \
  --train data/features/train.npz \
  --val data/features/val.npz \
  --n-estimators 300 \
  --output models/rf_ensemble
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

Place the PlantVillage dataset under `data/raw/train/`:
```
data/raw/train/
├── Pepper,_bell___Bacterial_spot/   (~1913 images)
├── Pepper,_bell___healthy/          (~1988 images)
├── Potato___Early_blight/           (~1939 images)
├── Potato___healthy/                (~1824 images)
├── Potato___Late_blight/            (~1939 images)
├── Tomato___Bacterial_spot/         (~1702 images)
├── Tomato___Early_blight/           (~1920 images)
├── Tomato___Late_blight/            (~1851 images)
├── Tomato___Septoria_leaf_spot/     (~1745 images)
└── Tomato___Target_Spot/            (~1827 images)
```

**Total: 10 classes, ~18,648 images.**

### 2. Run Preprocessing Pipeline

**Demo script (auto-detects demo_input/ or samples from dataset)**
```bash
# Process manually placed images:
cp path/to/your/images/*.jpg data/demo_input/
python scripts/demo_single_image.py

# Or sample from the dataset:
python scripts/demo_single_image.py -n 20
```

### 3. Create Splits
```bash
python scripts/02_make_splits.py --manifest data/processed/manifest.csv --output data/splits
```
Creates stratified train/validation/test splits.

### 4. Extract Features (for traditional ML)
```bash
python scripts/extract_features_dataset.py
```

### 5. Train Models
```bash
# CNN (MobileNetV3-Small, 10 epochs)
python scripts/03_train_cnn.py --backbone mobilenet_v3_small --epochs 10 --batch-size 32 --data-dir data/raw/train

# Random Forest ensemble
python scripts/train_rf_cli.py --train data/features/train.npz --val data/features/val.npz
```

### 6. Evaluate
```bash
python scripts/04_evaluate_cnn.py
```

## Testing

Run the complete test suite:

```bash
# All tests (54 tests)
pytest tests/ -v

# Specific test file
pytest tests/test_classifiers.py -v    # All 6 traditional classifiers
pytest tests/test_cnn_baseline.py -v   # CNN model tests
pytest tests/test_rf_ensemble.py -v    # Random Forest ensemble tests
pytest tests/test_preprocess.py -v     # Preprocessing tests
pytest tests/test_splits.py -v         # Data split tests

# With coverage
pytest tests/ --cov=src/plantdisease
```

**Test Coverage (54 tests):**
- **Classifiers** (`test_classifiers.py`): All 6 traditional classifiers — RF, SVM, Logistic Regression, KNN, XGBoost, CatBoost
- **CNN Baseline** (`test_cnn_baseline.py`): Model initialization, forward pass, exports, training
- **Random Forest Ensemble** (`test_rf_ensemble.py`): Training, prediction, evaluation, save/load, feature importance
- **Preprocessing** (`test_preprocess.py`): All preprocessing filters and utilities
- **Data Splits** (`test_splits.py`): Train/val/test split validation

### Pipeline Visual Testing

```bash
# Drop images in data/demo_input/ for manual testing, or sample from dataset:
python scripts/demo_single_image.py
```

## Configuration

Edit `config.yaml` for project-wide settings:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  features_dir: data/features

model:
  num_classes: 10
  backbone: mobilenet_v3_small
  
training:
  batch_size: 32
  epochs: 10
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

- **Automated Preprocessing Pipeline**: rembg / U2-Net background removal, HSV shadow removal, HSV disease segmentation (yellow + brown + dark necrotic), severity quantification
- **7 Models**: CNN (MobileNetV3) + 6 traditional ML classifiers (RF, SVM, LogReg, KNN, XGBoost, CatBoost)
- **99.93% CNN Accuracy**: Transfer learning with MobileNetV3-Small achieves near-perfect classification
- **109-Feature Vector**: Disease + leaf Gabor texture, CIELAB + HSV colour, morphology for traditional ML
- **Demo Script**: Manual (`data/demo_input/`) or dataset-sampling mode — outputs 3×3 visualization grids
- **Mobile Export**: ONNX and TorchScript support
- **PlantVillage Dataset**: 10-class dataset (~18,648 images) for Solanaceae crops
- **Flexible Training**: CLI scripts with hyperparameter control
- **Comprehensive Testing**: 54 unit tests covering all models and preprocessing
- **Checkpoint Management**: Save/load training states
- **Uncertainty Quantification**: Confidence thresholds and top-k predictions
- **Disease Severity**: Automated percentage-based severity quantification

## Performance

### CNN (MobileNetV3-Small)
- **Validation accuracy: 99.93%** (2795/2797 correct)
- 10 epochs training on 15,851 training images, validated on 2,797
- Supports real-time inference on mobile devices via TorchScript
- Efficient backbone (~2.5M parameters)
- Uncertainty threshold for filtering low-confidence predictions

### Traditional ML Classifiers (109 hand-crafted features)
- **SVM (RBF): 88.50%** (best traditional model)
- XGBoost: 88.10%
- CatBoost: 87.50%
- Logistic Regression: 87.10%
- KNN (k=6, RF-weighted): 85.50%
- Random Forest: 85.10%
- All 6 classifiers ≥85% accuracy
- Evaluated on 5,000 sampled images with 80/20 stratified train/test split
- Interpretable feature importance analysis available

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages:

- PyTorch 2.0+
- torchvision 0.15+
- scikit-learn
- xgboost
- catboost
- rembg
- NumPy, Pandas, SciPy
- OpenCV, Pillow
- pytest
