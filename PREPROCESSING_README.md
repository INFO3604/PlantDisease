# Preprocessing Pipeline — Changes & Documentation

## Overview

This document describes the **preprocessing pipeline** implemented for Task 1 (Feature Extraction & XGBoost Baseline). The pipeline replaces the supervisor's GrabCut-based segmentation with a completely different approach using **CIELAB a\*-channel segmentation**, and runs disease detection on the **segmented leaf** (background removed) rather than the full image to eliminate shadow false-positives.

---

## Why the Change?

| Problem | Original Pipeline | New Pipeline |
|---------|-------------------|--------------|
| **Segmentation** | GrabCut (supervisor already implemented) | LAB a\*-channel (completely different method) |
| **Shadow false-positives** | Disease overlay on full image → shadows misclassified as disease | Disease overlay on **segmented leaf** (black background) → no shadows |
| **Dependencies** | Requires careful GrabCut initialisation, slow | Pure OpenCV, fast, no parameter tuning needed |

---

## Pipeline Steps

```
Input Image (BGR)
  │
  ├─ 1. Resize (Lanczos interpolation, 256×256)
  │
  ├─ 2. White Balance (Gray-World algorithm)
  │     Scales each channel so its mean equals the global mean.
  │     Corrects colour casts from different lighting conditions.
  │
  ├─ 3. Denoise (Bilateral Filter, d=9, σ_color=75, σ_space=75)
  │     Edge-preserving smoothing — keeps disease boundaries sharp
  │     while removing sensor noise from flat regions.
  │
  ├─ 4. Contrast Enhancement (AGCWD)
  │     Adaptive Gamma Correction with Weighting Distribution.
  │     Adjusts gamma curve based on luminance histogram PDF/CDF.
  │     Dark images get more brightening than already-bright ones.
  │
  ├─ 5. Leaf Segmentation (LAB a*-channel)
  │     Converts to CIELAB colour space.
  │     Inverts the a* channel (green = low a* → high after inversion).
  │     Otsu thresholds the inverted a* channel.
  │     Morphological open/close cleanup + largest connected component.
  │     → Produces binary leaf mask + segmented leaf (background = black).
  │
  ├─ 6. Disease Detection (Mahalanobis distance — ON SEGMENTED LEAF)
  │     Collects healthy green pixels as reference distribution.
  │     Computes Mahalanobis distance in LAB space for all leaf pixels.
  │     Pixels with distance > 2.5 (threshold) are classified as diseased.
  │     Morphological cleanup + re-apply leaf mask to prevent overflow.
  │     → Produces disease mask + severity percentage.
  │
  └─ 7. Disease Overlay (on segmented leaf)
        Red-tinted overlay on the segmented leaf image.
        Only leaf pixels can be highlighted — no shadow contamination.
```

---

## Method Selection: Why LAB a\*?

Three segmentation methods were implemented and tested on 14 PlantVillage images across 7 tomato disease classes:

| Method | Success Rate | Dependencies | Speed | Notes |
|--------|-------------|--------------|-------|-------|
| **ExG Color-Index** | 13/14 (93%) | OpenCV only | Fast | Failed on one Target Spot image (mask too small 4.2%) |
| **LAB a\*-channel** | **14/14 (100%)** | **OpenCV only** | **Fast** | **Most robust — chosen as primary method** |
| SLIC Superpixel | 14/14 (100%) | scikit-image | Slow | Slightly higher severity (over-detection tendency) |

**LAB a\*-channel was selected** because:
- **100% success rate** across all test images
- **No external dependencies** beyond OpenCV
- **Fastest execution** (no superpixel computation)
- **Theoretically sound**: CIELAB a\* directly encodes the green↔red opponent axis, making it an ideal single channel for separating green leaf tissue from non-green background
- **Minimal tuning needed**: Otsu auto-selects the threshold

---

## Files Changed

### New Files Created

| File | Purpose |
|------|---------|
| `src/plantdisease/data/preprocess/leaf_segmentation.py` | Three leaf segmentation classes (ColorIndexSegmenter, LABSegmenter, SLICSegmenter) |
| `src/plantdisease/data/preprocess/pipeline.py` | Complete end-to-end pipeline: `PreprocessingPipeline` class with `run(image)` method |
| `scripts/test_pipeline.py` | Visual test script — runs pipeline on PlantVillage images, saves comparison grids |
| `PREPROCESSING_README.md` | This document |

### Files Modified

| File | Change |
|------|--------|
| `src/plantdisease/data/preprocess/__init__.py` | Added exports for `PreprocessingPipeline`, `PipelineResult`, `LABSegmenter`, `ColorIndexSegmenter`, `SLICSegmenter`, `SegmentationMethod` |

### Files Retained (Unchanged)

The original `segmentation.py` (GrabCut + Otsu) and `disease.py` (HSV-based) are **kept as-is** for backwards compatibility. The new pipeline does not depend on them.

---

## Test Results

Tested on 14 PlantVillage images (2 per disease class), disease detection on **segmented leaf**:

| # | Disease Class | Seg OK | Leaf Mask % | Severity % |
|---|--------------|--------|-------------|------------|
| 1 | Bacterial spot | YES | 43.5% | 10.1% |
| 2 | Bacterial spot | YES | 45.4% | 9.1% |
| 3 | Early blight | YES | 35.0% | 11.5% |
| 4 | Early blight | YES | 34.3% | 8.1% |
| 5 | Late blight | YES | 18.3% | 7.2% |
| 6 | Late blight | YES | 42.5% | 15.5% |
| 7 | Leaf Mold | YES | 39.8% | 8.2% |
| 8 | Leaf Mold | YES | 29.4% | 8.3% |
| 9 | Septoria leaf spot | YES | 17.1% | 3.4% |
| 10 | Septoria leaf spot | YES | 40.6% | 8.0% |
| 11 | Spider mites | YES | 32.3% | 8.8% |
| 12 | Spider mites | YES | 39.5% | 5.0% |
| 13 | Target Spot | YES | 42.2% | 10.1% |
| 14 | Target Spot | YES | 27.3% | 11.6% |

**14/14 images segmented successfully (100%)**. Severity percentages are reasonable (3–16%), with no shadow false-positives.

Visual output grids are saved in `data/preprocessed_output/pipeline_test/`.

---

## Usage

### Quick Start

```python
import cv2
from plantdisease.data.preprocess import PreprocessingPipeline

# Create pipeline (LAB a* is the default)
pipe = PreprocessingPipeline()

# Run on an image
image = cv2.imread("path/to/leaf.jpg")
result = pipe.run(image)

# Access outputs
print(f"Segmentation: {result.segmentation_method}")
print(f"Leaf mask ratio: {result.mask_ratio:.1%}")
print(f"Disease severity: {result.severity_percent:.1f}%")

# Save outputs
cv2.imwrite("segmented.jpg", result.segmented_leaf)
cv2.imwrite("disease_overlay.jpg", result.disease_overlay)
cv2.imwrite("leaf_mask.png", result.leaf_mask)
```

### Run Visual Test

```bash
python scripts/test_pipeline.py
```

This produces per-image 6-panel grids (Original → Enhanced → Leaf Mask → Segmented Leaf → Disease Mask → Disease Overlay) and a combined summary grid.

### Pipeline Configuration

```python
pipe = PreprocessingPipeline(
    target_size=(256, 256),           # Resize dimensions
    segmentation_method="lab_astar",  # "lab_astar" | "color_index" | "slic_superpixel"
    color_index="exg",                # Only used if segmentation_method="color_index"
    disease_threshold=2.5,            # Mahalanobis threshold (lower = stricter)
)
```

---

## Key Design Decisions

1. **Disease detection on segmented leaf**: The segmented leaf has a black background. The Mahalanobis distance computation only considers pixels within the leaf mask, so background shadows, table edges, and other artefacts cannot be misclassified as disease.

2. **LAB colour space throughout**: Both leaf segmentation (a\* channel) and disease detection (Mahalanobis in LAB) operate in CIELAB, which is perceptually uniform and separates chrominance from luminance — making both steps robust to lighting variations.

3. **No GrabCut anywhere**: The supervisor's pipeline uses GrabCut. This pipeline is completely independent — different algorithm, different theory, different implementation.

4. **Morphological post-processing**: Both segmentation and disease detection include open/close operations and largest-component extraction with leaf-mask re-application to prevent mask overflow (a previously fixed bug).
