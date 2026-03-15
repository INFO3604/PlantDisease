# Preprocessing Pipeline Documentation

## Overview

Image preprocessing pipeline for automated plant disease detection on Solanaceae
(tomato, potato, bell pepper) leaf images from the PlantVillage dataset.

**Primary method**: Deep-learning background removal (rembg / U2-Net) with HSV-based
shadow removal, HSV disease segmentation (yellow chlorosis + brown necrosis), and
colour-coded disease visualisation.

All disease detection runs on the **segmented leaf** (background removed via rembg)
to eliminate background contamination and shadow false-positives.

---

## Pipeline Steps

```
Input Image (BGR)
  │
  ├── 1. Remove Background  (rembg / U2-Net deep-learning model)
  │      Produces RGBA image with transparent background.
  │      Leaf mask derived from the alpha channel (α > 127).
  │
  ├── 2. Resize  (Lanczos-4 interpolation, 300×300)
  │      Operates on RGBA to preserve alpha channel.
  │      Preserves fine lesion edges and vein texture.
  │
  ├── 3. Shadow Removal  (HSV colour-space thresholds)
  │      Identifies shadow pixels: V < 80 AND S < 50.
  │      Replaces shadow pixels with local mean, restricted to leaf mask.
  │      Produces shadow-removed image + binary shadow mask.
  │      Effective leaf mask refined by excluding shadow pixels.
  │
  ├── 4. HSV Disease Segmentation  (yellow / brown thresholds)
  │      Yellow (chlorosis):  H ∈ [15, 40], S ∈ [40, 255], V ∈ [50, 255]
  │      Brown (necrosis):    H ∈ [0, 25],  S ∈ [30, 255], V ∈ [30, 220]
  │      Reddish-brown:       H ∈ [165, 180], S ∈ [30, 255], V ∈ [30, 220]
  │      Dark necrotic:       H ∈ [0, 30],  S ∈ [20, 200], V ∈ [10, 60]
  │      Adjacent green regions near lesions optionally included.
  │      Morphological cleanup (open → close → contour filtering).
  │      Leaf mask re-applied after cleanup to prevent pixel leakage.
  │      → Disease mask + yellow mask + brown mask + severity metrics
  │
  ├── 5. Severity Calculation
  │      severity_percent = (diseased_pixels / total_leaf_pixels) × 100
  │      Separate yellow and brown severity percentages also computed.
  │
  └── 6. Data Normalisation  (pixel values scaled to [0, 1])
         float32 output suitable for model input.
```

---

## Background Removal: rembg / U2-Net

The pipeline uses the `rembg` library with the U2-Net model for background removal.
This deep-learning approach provides accurate foreground segmentation for plant leaves
against varied backgrounds, producing an RGBA image where the alpha channel encodes
the leaf boundary.

### How It Works

1. Input BGR image is converted to RGB for rembg processing
2. rembg runs the U2-Net salient object detection model
3. Output is an RGBA image with transparent background
4. The leaf mask is derived by thresholding the alpha channel (α > 127)

### Advantages

- Works on any background colour (plain, complex, natural)
- Handles overlapping leaves and irregular shapes
- No manual threshold tuning required
- Robust across different lighting conditions

---

## Shadow Removal: HSV Thresholds

After background removal, shadows on the leaf surface are identified and corrected
using HSV colour-space analysis.

### Detection Criteria

A pixel is classified as a shadow if **both conditions** are met:
- Value (V) < 80 — dark pixel
- Saturation (S) < 50 — desaturated pixel

### Correction

Shadow pixels are replaced with the local mean of non-shadow neighbouring pixels,
restricted to the leaf mask region. This preserves disease regions while removing
shadow artefacts.

---

## Disease Detection: HSV Segmentation

The `DiseaseSegmenter` class detects diseased regions using multiple HSV colour ranges
applied to the shadow-removed leaf image.

### Detection Ranges

| Symptom Type | Hue (H) | Saturation (S) | Value (V) | Clinical Meaning |
|-------------|---------|-----------------|-----------|-----------------|
| Yellow (chlorosis) | 15–40 | 40–255 | 50–255 | Chlorosis, nutrient deficiency, early blight |
| Brown (necrosis) | 0–25 | 30–255 | 30–220 | Necrosis, rot, late blight lesions |
| Reddish-brown | 165–180 | 30–255 | 30–220 | Hue wraparound for red-brown tones |
| Dark necrotic | 0–30 | 20–200 | 10–60 | Dead tissue, advanced necrosis |

### Adjacent Green Detection

Green regions immediately adjacent to disease lesions are optionally included in the
disease mask. These represent transition zones where disease is actively spreading.
Detection uses a dilation-based approach: the disease mask is dilated with an elliptical
kernel, and the intersection with the green mask identifies transitional tissue.

### Morphological Cleanup

1. **Opening** (erosion → dilation): removes small noise specks
2. **Closing** (dilation → erosion): fills small holes within disease regions
3. **Contour filtering**: removes contours smaller than 20 pixels (min_contour_area)
4. **Leaf mask re-application**: ensures no pixels leak outside the leaf boundary after morphological operations

### Severity Calculation

```
severity_percent = (diseased_pixels / total_leaf_pixels) × 100
```

Severity is also broken down by symptom type (yellow vs brown).

| Severity Level | Range |
|---------------|-------|
| Healthy / Trace | < 5% |
| Low | 5–15% |
| Moderate | 15–30% |
| Severe | 30–50% |
| Very Severe | > 50% |

---

## Disease Overlay Visualisation

| Category | Overlay Colour (BGR) | Clinical Meaning |
|----------|---------------------|-----------------|
| Yellow (chlorosis) | Yellow (0, 255, 255) | Chlorosis, nutrient deficiency |
| Brown (necrosis) | Red (0, 0, 255) | Necrotic tissue, blight lesions |

Alpha blending: 0.5 (overlay + original texture visible).

---

## Demo Visualisation Grid

The demo script produces a **3×3 visualization grid** per image:

| | Column 1 | Column 2 | Column 3 |
|---|----------|----------|----------|
| **Row 1** | Original | BG Removed (rembg) | Shadow Removed |
| **Row 2** | Yellow Regions | Brown Regions | Disease Mask (combined) |
| **Row 3** | Disease Overlay | Leaf Mask (from alpha) | Shadow Mask |

Each panel is 300×300 pixels with colour-coded labels indicating the pipeline step.

---

## Scripts & Commands

### Demo Script

```bash
# Process all images in a folder:
python scripts/demo_single_image.py --input data/demo_input --output data/demo_output/rembg_run

# Process a single image:
python scripts/demo_single_image.py --input path/to/leaf.jpg --output data/demo_output/rembg_run
```

### Python API

```python
import cv2
from plantdisease.data.preprocess import PreprocessingPipeline

pipe = PreprocessingPipeline()
image = cv2.imread("path/to/leaf.jpg")
result = pipe.run(image)

# Access outputs
print(f"Severity: {result.severity_percent:.1f}%")
print(f"Diseased pixels: {result.diseased_pixels}")
print(f"Total leaf pixels: {result.total_leaf_pixels}")
print(f"Yellow pixels: {result.yellow_pixels}")
print(f"Brown pixels: {result.brown_pixels}")
print(f"Steps applied: {result.steps_applied}")

cv2.imwrite("overlay.jpg", result.disease_overlay)
```

### Pipeline Configuration

```python
pipe = PreprocessingPipeline(
    target_size=(300, 300),   # Resize dimensions (default 300×300)
    normalize=True,           # Scale pixel values to [0, 1]
)
```

---

## Project Files

### Pipeline Source Code

| File | Purpose |
|------|---------|
| `src/plantdisease/data/preprocess/pipeline.py` | `PreprocessingPipeline` — full end-to-end pipeline with `run(image)` method |
| `src/plantdisease/data/preprocess/background.py` | `remove_background_rembg()` — deep-learning background removal |
| `src/plantdisease/data/preprocess/shadow.py` | `remove_shadows_hsv_threshold()` — HSV-based shadow detection and removal |
| `src/plantdisease/data/preprocess/disease.py` | `DiseaseSegmenter` — HSV disease segmentation + severity metrics |
| `src/plantdisease/data/preprocess/__init__.py` | Clean exports for all pipeline classes |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/demo_single_image.py` | Demo — processes images, outputs 3×3 visualization grids |

---

## Tuning Guide

| Problem | Adjustment |
|---------|-----------|
| Missing yellow/chlorotic spots | Lower `DEFAULT_YELLOW_RANGE.s_min` or `v_min` |
| Missing brown/necrotic spots | Lower `DEFAULT_BROWN_RANGE.s_min` or `v_min` |
| Too many false positives in disease mask | Raise saturation/value minimums in HSV ranges |
| Small spots being ignored | Decrease `min_contour_area` (default: 20) |
| Shadows still affecting detection | Adjust shadow thresholds in `remove_shadows_hsv_threshold()` |
| Disease severity > 100% | Ensure leaf mask is re-applied after morphological cleanup |

---

## Key Design Decisions

1. **Deep-learning background removal (rembg)**: U2-Net provides robust foreground segmentation that works on any background without manual threshold tuning. The alpha channel directly provides the leaf mask.

2. **HSV disease segmentation**: Multiple HSV ranges target specific symptom types (chlorosis, necrosis, dark necrotic spots). Widened thresholds with reddish-brown wraparound detection ensure comprehensive coverage.

3. **Shadow removal before disease detection**: HSV-based shadow identification and correction prevents dark shadows from being misclassified as brown/necrotic disease regions.

4. **Post-cleanup leaf mask re-application**: Morphological operations (opening, closing) can expand masks beyond the leaf boundary. Re-applying the leaf mask after cleanup prevents pixel leakage and ensures severity stays within 0–100%.

5. **Adjacent green inclusion**: Dilation-based detection of green tissue adjacent to disease lesions captures transition zones where infection is actively spreading, improving severity accuracy.

6. **Normalisation for model input**: Final float32 [0, 1] scaling provides standardised input for downstream CNN or XGBoost models.
