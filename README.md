# Plant Disease Detection

A machine learning project for automated plant disease detection and classification on Solanaceae (tomato) crops using the PlantVillage dataset. Combines a custom image preprocessing pipeline with transfer learning (CNN) and Random Forest ensemble baselines.

## Project Overview

This project implements:

1. **Image Preprocessing Pipeline** - Automated leaf segmentation, disease detection & severity quantification
2. **CNN Baseline** - Deep learning models using MobileNetV3 and EfficientNet
3. **Random Forest Ensemble** - Ensemble classifier with Gabor texture, CIELAB colour, and morphological features

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
│   │       ├── background.py       # rembg deep-learning background removal
│   │       ├── shadow.py           # HSV shadow detection and removal
│   │       ├── disease.py          # HSV disease segmentation + severity
│   │       ├── denoise.py          # Denoising utilities
│   │       ├── grayscale.py        # Grayscale conversion utilities
│   │       └── resize_standardize.py # Resize and standardization
│   ├── features/                   # Feature extraction
│   │   └── extract_features.py     # 55-feature extractor (Gabor, CIELAB, morphology)
│   ├── models/                     # Model implementations
│   │   ├── cnn_baseline.py         # CNN models (MobileNetV3, EfficientNet)
│   │   ├── rf_ensemble.py          # Random Forest ensemble classifier
│   │   ├── evaluate.py             # Evaluation metrics
│   │   └── train.py                # Training loops
│   └── utils/                      # Utilities (logging, paths)
├── scripts/                        # Command-line scripts
│   ├── demo_single_image.py        # Demo: process images & output 3×3 visualization grids
│   ├── train_cnn_cli.py            # CNN training CLI
│   ├── train_rf_cli.py             # Random Forest training CLI
│   ├── evaluate_cli.py             # Model evaluation CLI
│   ├── 02_make_splits.py           # Create train/val/test splits
│   ├── 03_train_cnn.py             # Train CNN models
│   ├── 04_evaluate_cnn.py          # Evaluate CNN models
│   └── 05_inference_cnn.py         # CNN inference
├── tests/                          # Test suite
│   ├── test_preprocess.py          # Preprocessing tests
│   ├── test_cnn_baseline.py        # CNN tests
│   ├── test_rf_ensemble.py         # Random Forest ensemble tests
│   └── test_splits.py              # Data split tests
├── data/                           # Data directory
│   ├── New Plant Diseases Dataset(Augmented)/
│   │   └── train/                  # PlantVillage augmented dataset (12 classes)
│   │       ├── Pepper,_bell___Bacterial_spot/
│   │       ├── Pepper,_bell___healthy/
│   │       ├── Potato___Early_blight/
│   │       ├── Potato___healthy/
│   │       ├── Potato___Late_blight/
│   │       ├── Strawberry___healthy/
│   │       ├── Strawberry___Leaf_scorch/
│   │       ├── Tomato___Bacterial_spot/
│   │       ├── Tomato___Early_blight/
│   │       ├── Tomato___Late_blight/
│   │       ├── Tomato___Septoria_leaf_spot/
│   │       └── Tomato___Target_Spot/
│   ├── demo_input/                 # Drop images here for manual demo processing
│   ├── demo_output/                # Demo visualization output
│   │   └── rembg_run/              # Grid images + features.csv
│   └── features/                   # Extracted features (.npz)
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
├── PREPROCESSING_README.md         # Detailed preprocessing pipeline docs
└── Methodology_Preprocessing_Pipeline.docx  # Formal methodology write-up
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

Both modes save a 3×3 visualization grid per image and export 55 features + class labels to `features.csv`.

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

### Random Forest Ensemble (`src/plantdisease/models/rf_ensemble.py`)

Random Forest ensemble classifier using the 55-feature vector from
`src/plantdisease/features/extract_features.py`.

**Features (55-dim total):**
- Gabor texture: 36 dimensions (12 filter banks × 3 statistics)
- CIELAB colour: 6 dimensions (L*, a*, b* mean + std)
- Severity ratios: 3 dimensions (disease, yellow, brown)
- Morphology: 10 dimensions (lesion count, area, perimeter, shape)

**Configuration:**
- Trees: 300
- Class weighting: balanced
- Max features: sqrt
- Min samples split: 5

**Usage:**
```python
from src.plantdisease.models.rf_ensemble import RFEnsembleClassifier

# Train
classifier = RFEnsembleClassifier(n_estimators=300)
classifier.fit(X_train, y_train, X_val, y_val, feature_names=names)

# Predict
predictions = classifier.predict(X_test)
```

## CLI Scripts

### 1. CNN Training (`scripts/train_cnn_cli.py`)

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

### 2. Random Forest Training (`scripts/train_rf_cli.py`)

Train Random Forest ensemble:

```bash
python scripts/train_rf_cli.py \
  --train data/features/train.npz \
  --val data/features/val.npz \
  --n-estimators 300 \
  --output models/rf_ensemble
```

### 3. Model Evaluation (`scripts/evaluate_cli.py`)

Evaluate trained models:

```bash
python scripts/evaluate_cli.py \
  --model-path models/checkpoints/best_model.pt \
  --test-data data/processed \
  --model-type cnn
```

## Data Processing Pipeline

### 1. Prepare Your Data

Place the PlantVillage dataset under `data/`:
```
data/New Plant Diseases Dataset(Augmented)/
└── train/
    ├── Pepper,_bell___Bacterial_spot/   (~1913 images)
    ├── Pepper,_bell___healthy/          (~1988 images)
    ├── Potato___Early_blight/           (~1939 images)
    ├── Potato___healthy/                (~1824 images)
    ├── Potato___Late_blight/            (~1939 images)
    ├── Strawberry___healthy/            (~1824 images)
    ├── Strawberry___Leaf_scorch/        (~1774 images)
    ├── Tomato___Bacterial_spot/         (~1702 images)
    ├── Tomato___Early_blight/           (~1920 images)
    ├── Tomato___Late_blight/            (~1851 images)
    ├── Tomato___Septoria_leaf_spot/     (~1745 images)
    └── Tomato___Target_Spot/            (~1827 images)
```

Or set the `DATASET_DIR` environment variable to point to the `train/` folder elsewhere.

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

### 4. Train Models
```bash
# CNN
python scripts/03_train_cnn.py

# Random Forest ensemble
python scripts/train_rf_cli.py --train data/features/train.npz --val data/features/val.npz
```

### 5. Evaluate
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
pytest tests/test_rf_ensemble.py -v
pytest tests/test_preprocess.py -v
pytest tests/test_splits.py -v

# With coverage
pytest tests/ --cov=src/plantdisease
```

**Test Coverage:**
- CNN Baseline: model initialization, forward pass, exports, training
- Random Forest Ensemble: training, prediction, evaluation, save/load, feature importance
- Preprocessing: all preprocessing filters and utilities
- Data Splits: train/val/test split validation

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

- **Automated Preprocessing Pipeline**: rembg / U2-Net background removal, HSV shadow removal, HSV disease segmentation (yellow + brown + dark necrotic), severity quantification
- **Multiple Backends**: MobileNetV3, EfficientNet, Random Forest ensemble
- **Demo Script**: Manual (`data/demo_input/`) or dataset-sampling mode — outputs 3×3 visualization grids
- **Mobile Export**: ONNX and TorchScript support
- **PlantVillage Dataset**: 12-class augmented dataset (~22K images) for Solanaceae crops
- **Flexible Training**: CLI scripts with hyperparameter control
- **Comprehensive Testing**: Unit tests + stratified-sample visual pipeline demo
- **Checkpoint Management**: Save/load training states
- **Uncertainty Quantification**: Confidence thresholds and top-k predictions
- **Disease Severity**: Automated percentage-based severity quantification

## Performance

Both models achieve strong performance on the plant disease classification task:

**CNN Baseline:**
- Supports real-time inference on mobile devices via TorchScript
- Efficient backbone selection (MobileNetV3-Small ~2.5M params)
- Uncertainty threshold for filtering low-confidence predictions

**Random Forest Ensemble:**
- Fast training and inference
- Interpretable Gini feature importance
- 55-feature vector: Gabor texture + CIELAB colour + morphology

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages:

- rembg
- PyTorch 1.10+
- torchvision 0.11+
- scikit-learn
- NumPy, Pandas, SciPy
- OpenCV, Pillow
- pytest
