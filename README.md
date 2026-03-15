# Plant Disease Detection

A machine learning project for automated plant disease detection and classification on Solanaceae (tomato) crops using the PlantVillage dataset. Combines a custom image preprocessing pipeline with transfer learning (CNN) and gradient boosting (XGBoost) baselines.

## Project Overview

This project implements:

1. **Image Preprocessing Pipeline** - Automated leaf segmentation, disease detection & severity quantification
2. **CNN Baseline** - Deep learning models using MobileNetV3 and EfficientNet
3. **XGBoost Baseline** - Gradient boosting with hand-crafted image features (HSV, LBP, GLCM)

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
│   │   ├── download.py             # Dataset download utilities
│   │   ├── ingestion.py            # Data loading
│   │   ├── splits.py               # Train/val/test splitting
│   │   └── preprocess/             # Image preprocessing
│   │       ├── pipeline.py         # PreprocessingPipeline (main entry point)
│   │       ├── leaf_segmentation.py # WatershedSegmenter, LABSegmenter, ColorIndexSegmenter, SLICSegmenter
│   │       ├── augment.py          # Data augmentation
│   │       ├── contrast.py         # Contrast enhancement (CLAHE, histogram)
│   │       ├── denoise.py          # Denoising (bilateral, median, gaussian)
│   │       ├── disease.py          # HSV-based disease detection (legacy)
│   │       ├── grayscale.py        # Grayscale conversion
│   │       ├── resize_standardize.py # Resize and standardization
│   │       └── ...                 # Additional preprocessing modules
│   ├── models/                     # Model implementations
│   │   ├── cnn_baseline.py         # CNN models (MobileNetV3, EfficientNet)
│   │   ├── xgboost_baseline.py     # XGBoost classifier
│   │   ├── evaluate.py             # Evaluation metrics
│   │   ├── features.py             # Feature extraction (HSV, LBP, GLCM)
│   │   └── train.py                # Training loops
│   └── utils/                      # Utilities (logging, paths)
├── scripts/                        # Command-line scripts
│   ├── demo_single_image.py        # Demo: process images & output visualization grids
│   ├── test_pipeline.py            # Full pipeline test (28 images, 7 classes)
│   ├── preprocess_cli.py           # Batch preprocessing CLI
│   ├── train_cnn_cli.py            # CNN training CLI
│   ├── train_xgb_cli.py            # XGBoost training CLI
│   ├── evaluate_cli.py             # Model evaluation CLI
│   ├── 00_download_data.py         # Download dataset
│   ├── 01_preprocess_images.py     # Batch preprocessing
│   ├── 02_make_splits.py           # Create train/val/test splits
│   ├── 03_train_cnn.py             # Train CNN models
│   ├── 04_evaluate_cnn.py          # Evaluate CNN models
│   ├── 04_evaluate.py              # Generic evaluation
│   └── 05_inference_cnn.py         # CNN inference
├── tests/                          # Test suite
│   ├── test_preprocess.py          # Preprocessing tests (21 tests)
│   ├── test_cnn_baseline.py        # CNN tests (17 tests)
│   ├── test_task1_xgboost.py       # XGBoost tests (18 tests)
│   └── test_splits.py              # Data split tests (3 tests)
├── data/                           # Data directory
│   ├── demo_input/                 # Place images here for demo
│   ├── demo_output/                # Demo visualization output
│   ├── preprocessed_output/        # Pipeline test outputs
│   │   └── pipeline_test/          # Grid images from test_pipeline.py
│   └── features/                   # Extracted features (.npz)
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
├── PREPROCESSING_README.md         # Detailed preprocessing pipeline docs
└── Methodology_Preprocessing_Pipeline.docx  # Formal methodology write-up
```

## Image Preprocessing Pipeline

The core preprocessing pipeline (`src/plantdisease/data/preprocess/pipeline.py`) provides automated leaf segmentation, disease detection, and severity quantification. It uses **adaptive Watershed segmentation** for robust leaf isolation and **fused disease detection** (Watershed markers + Mahalanobis distance) with texture-aware shadow rejection.

### Pipeline Stages

| Step | Method | Purpose |
|------|--------|---------|
| 1. Resize | Lanczos-4 (256×256) | Spatial standardisation |
| 2. White Balance | Gray-World | Illumination normalisation |
| 3. Denoise | Bilateral Filter (d=9) | Edge-preserving noise removal |
| 4. Contrast | AGCWD | Adaptive gamma correction |
| 5. Segment | Adaptive Watershed (MAD background model) | Leaf isolation (green, brown, dried, dark tissue) |
| 6. Disease | Fused Watershed + Mahalanobis with texture-aware shadow rejection | Disease detection on segmented leaf |
| 7. Overlay | Colour-coded 6-category (alpha=0.75) | Disease severity visualisation |

### Quick Demo

```bash
# Place leaf images in data/demo_input/, then run:
python scripts/demo_single_image.py --input data/demo_input --output data/demo_output
```

This processes **all** images in the input folder and saves a 4×3 visualization grid per image showing every pipeline stage + severity analysis.

### Full Pipeline Test (28 images, 7 disease classes)

```bash
python scripts/test_pipeline.py
```

Results: **28/28 images segmented successfully (100%)**.

### Python API

```python
import cv2
from plantdisease.data.preprocess import PreprocessingPipeline

pipe = PreprocessingPipeline()
image = cv2.imread("path/to/leaf.jpg")
result = pipe.run(image)

print(f"Severity: {result.severity_percent:.1f}%")
cv2.imwrite("overlay.jpg", result.disease_overlay)
```

See [PREPROCESSING_README.md](PREPROCESSING_README.md) for full documentation including design decisions, tuning guide, and overlay colour specifications.

### SAM (Segment Anything Model)

The pipeline also supports an optional SAM segmenter (`sam_segmenation.py`):
- Uses a local SAM checkpoint at `models/sam_vit_b.pth` if available.
- Set the `SAM_CHECKPOINT` environment variable to use a custom path.
- On machines without GPU, the loader uses a CPU-friendly configuration; CUDA is recommended for best performance.

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
│   └── ...
├── class2/
│   └── ...
└── class3/
    └── ...
```

### 2. Run Preprocessing Pipeline

**Option A: Demo script (recommended for single images / presentations)**
```bash
python scripts/demo_single_image.py --input data/demo_input --output data/demo_output
```

**Option B: Full test suite (28 PlantVillage images, 7 disease classes)**
```bash
python scripts/test_pipeline.py
```

**Option C: Batch preprocessing CLI**
```bash
python scripts/preprocess_cli.py --input data/raw --output data/processed --denoise bilateral --contrast clahe
```

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
- **Total: 61 tests - All passing**

### Pipeline Visual Testing

```bash
# Full visual test: 28 images across 7 tomato disease classes
python scripts/test_pipeline.py

# Demo: process any images you drop into a folder
python scripts/demo_single_image.py --input data/demo_input --output data/demo_output
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

- **Automated Preprocessing Pipeline**: Adaptive Watershed leaf segmentation, fused disease detection (Watershed + Mahalanobis), texture-aware shadow rejection, 6-category colour-coded overlay
- **Multiple Backends**: MobileNetV3, EfficientNet, XGBoost
- **Demo Script**: Process any images with `demo_single_image.py` — outputs 4×3 visualization grids
- **Mobile Export**: ONNX and TorchScript support
- **Data Augmentation**: Extensive preprocessing pipeline
- **Flexible Training**: CLI scripts with hyperparameter control
- **Comprehensive Testing**: 61 unit tests + 28-image visual pipeline test (100% pass rate)
- **Checkpoint Management**: Save/load training states
- **Uncertainty Quantification**: Confidence thresholds and top-k predictions
- **Disease Severity**: Automated percentage-based severity quantification

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
