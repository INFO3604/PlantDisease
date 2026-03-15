# Preprocessing Pipeline Documentation

## Overview

Image preprocessing pipeline for automated plant disease detection on Solanaceae
(tomato, potato, bell pepper) leaf images from the PlantVillage dataset.

**Primary method**: Adaptive Watershed leaf segmentation with fused Watershed + Mahalanobis disease detection, texture-aware shadow suppression, and 6-category colour-coded disease visualisation.

All disease detection runs on the **segmented leaf** (background removed) to eliminate shadow false-positives.

---

## Pipeline Steps

```
Input Image (BGR)
  |
  +-- 1. Resize (Lanczos-4 interpolation, 256x256)
  |      Preserves fine lesion edges and vein texture.
  |
  +-- 2. White Balance (Gray-World algorithm)
  |      Scales each BGR channel so its mean equals the global mean.
  |      Corrects colour casts from different lighting/cameras.
  |
  +-- 3. Denoise (Bilateral Filter, d=9, sigma_c=75, sigma_s=75)
  |      Edge-preserving smoothing — keeps disease boundaries sharp
  |      while removing sensor noise from flat regions.
  |
  +-- 4. Contrast Enhancement (AGCWD)
  |      Adaptive Gamma Correction with Weighting Distribution.
  |      Per-intensity adaptive gamma based on luminance PDF/CDF.
  |      Dark images brightened more; bright images left alone.
  |
  +-- 5. Leaf Segmentation (Adaptive Watershed — 13-step algorithm)
  |      Step 1:  Colour-space conversions (LAB, HSV, gray)
  |      Step 2:  Robust border-sampled background model (MAD outlier rejection)
  |      Step 3:  Texture map (local std-dev — leaf has microtexture, shadows smooth)
  |      Step 4:  Dark-tissue boost (near-black necrosis always kept)
  |      Step 5:  Combined score = MD + texture_weight * texture + dark_boost
  |      Step 6:  Adaptive Otsu threshold on score histogram
  |      Step 7:  Morphological cleanup + largest connected component
  |      Step 8:  Conditional shadow suppression (V>80 guard protects dark tissue)
  |      Step 9:  Interior hole fill (pre-Watershed)
  |      Step 10: Watershed markers from distance transform
  |      Step 11: cv2.watershed() boundary refinement
  |      Step 12: Interior hole fill (post-Watershed)
  |      → Binary leaf mask + segmented leaf (background = black)
  |
  +-- 6. Disease Detection (Fused: Watershed markers + Mahalanobis distance)
  |      Method A — Watershed markers:
  |        Seed markers from HSV Hue (warm tones) and LAB a* (red/brown shift).
  |        Marker-controlled watershed inside the leaf mask.
  |      Method B — Mahalanobis distance:
  |        Healthy green pixels define reference distribution (mean + covariance).
  |        Pixels with Mahalanobis distance > threshold classified as diseased.
  |        Texture-aware shadow rejection (low texture + dark + low saturation = shadow).
  |      Fusion: Union of both methods → morphological cleanup.
  |      → Disease mask + severity percentage
  |
  +-- 7. Colour-Coded Disease Overlay (alpha=0.75)
         6 categories based on HSV/LAB colour of diseased pixels:
           Dark necrotic    → Deep Red
           Brown / blight   → Bright Orange
           Light brown / tan → Warm Amber
           Yellow chlorotic  → Bright Yellow
           Red / purple spot → Magenta
           Other diseased    → Dark Crimson
         White contour outlines drawn around each diseased region.
```

---

## Leaf Segmentation: Adaptive Watershed

The Watershed segmenter addresses 6 major failure modes encountered during development
and has been validated on **20+ disease types** across tomato, potato, and bell pepper.

### Failure Modes Addressed

| # | Problem | Solution |
|---|---------|----------|
| 1 | Dark/necrotic tissue erased | Mahalanobis distance in LAB (no hue ranges) + dark_tissue_boost (V≤40 → +8 score) |
| 2 | Shadows included in mask | Texture map (leaf has microtexture, shadows smooth) + conditional shadow suppression (V>80 guard) |
| 3 | Under-segmentation (leaf ≈ bg colour) | Adaptive Otsu on score histogram (self-calibrates per image) |
| 4 | Border contamination in bg model | MAD-based outlier rejection on border pixels |
| 5 | Chromatic backgrounds (blue/purple) | Mahalanobis in LAB handles any background colour |
| 6 | Holes in mask (lesion pits, crevices) | _fill_holes() applied twice: pre-Watershed and post-Watershed |

### Shadow Suppression Details

The texture-aware shadow suppression fires **only when coverage > 65%** (configurable).
It removes pixels satisfying ALL of:
- Statistically close to background (MD < robust_shadow_cut)
- Bright (V > 80) — dark necrotic tissue is **never** removed
- Currently in the rough mask

This replaces the simpler fixed-threshold HSV shadow rejection (S<70 & V<85) from the
previous pipeline version.

### Legacy Methods

Three legacy segmentation methods remain available for comparison:

| Method | Status | Notes |
|--------|--------|-------|
| **Watershed** | **Primary (default)** | Adaptive, handles dark tissue + shadows |
| LAB a*-channel | Legacy | Triple-channel (a* + chroma + b*), good for green leaves |
| Color-Index (ExG) | Legacy | Vegetation indices, fast but misses brown tissue |
| SLIC Superpixel | Legacy | Requires cv2.ximgproc, slow |

---

## Disease Detection: Fused Approach

The fused disease detector combines two independent methods for maximum recall:

### Method A: Watershed Markers (Hue + a*)
- Seed foreground markers from warm hue (H 4–38, S≥28) OR high a* (≥68th percentile)
- Seed background markers from low a* (≤42nd percentile) AND non-warm hue
- Marker-controlled watershed inside the leaf mask
- Best for: blight lesions, bacterial spots with distinct colour shift

### Method B: Mahalanobis Distance
- Healthy green pixels define the reference distribution
- Pixels with Mahalanobis distance > 2.5 classified as diseased
- Texture-aware shadow rejection (replaces fixed HSV gates):
  - Low texture (std-dev < 5) + dark (V < 100) + low saturation (S < 80) = shadow → removed
  - Very dark + low texture (V < 35, texture < 3) = deep shadow → removed
- Best for: subtle discolouration, early-stage disease

### Fusion
Union of both methods → morphological close (fill gaps) → open (remove noise) → re-apply leaf mask.

---

## Disease Overlay Colours

| Category | Criteria | Overlay Colour (BGR) | Clinical Meaning |
|----------|----------|---------------------|-----------------|
| Dark necrotic | H<22 & V≤80; or H<15 & S≤50 | Deep Red (30,30,255) | Dead tissue, advanced necrosis |
| Brown / blight | H∈[8,22), S>25, V>80 | Bright Orange (0,140,255) | Early browning, lesion margins |
| Yellow / chlorotic | H∈[15,35], S>40 | Bright Yellow (0,255,255) | Chlorosis, nutrient deficiency |
| Red / purple spot | LAB a*>145, S>50 | Magenta (180,50,200) | Anthocyanin accumulation |
| Other diseased | None of the above | Dark Crimson (30,0,180) | Atypical discolouration |

Alpha blending: 0.75 (bold overlay + original texture visible). White contour outlines around each region.

---

## Scripts & Commands

### Demo Script

```bash
# Process all images in a folder:
python scripts/demo_single_image.py --input data/demo_input --output data/demo_output

# Process a single image:
python scripts/demo_single_image.py --input path/to/leaf.jpg --output data/demo_output
```

### Full Test Suite (28 images, 7 disease classes)

```bash
python scripts/test_pipeline.py
```

### Python API

```python
import cv2
from plantdisease.data.preprocess import PreprocessingPipeline

# Create pipeline (Watershed is default)
pipe = PreprocessingPipeline()

# Run on an image
image = cv2.imread("path/to/leaf.jpg")
result = pipe.run(image)

# Access outputs
print(f"Segmentation: {result.segmentation_method}")
print(f"Success: {result.segmentation_success}")
print(f"Leaf mask ratio: {result.mask_ratio:.1%}")
print(f"Disease severity: {result.severity_percent:.1f}%")
print(f"Steps applied: {result.steps_applied}")

cv2.imwrite("segmented.jpg", result.segmented_leaf)
cv2.imwrite("disease_overlay.jpg", result.disease_overlay)
```

### Pipeline Configuration

```python
pipe = PreprocessingPipeline(
    target_size=(256, 256),            # Resize dimensions
    segmentation_method="watershed",   # "watershed" | "lab_astar" | "color_index" | "slic_superpixel"
    disease_threshold=2.5,             # Mahalanobis threshold (lower = stricter)
)
```

---

## Project Files

### Pipeline Source Code

| File | Purpose |
|------|---------|
| `src/plantdisease/data/preprocess/pipeline.py` | `PreprocessingPipeline` — full end-to-end pipeline with `run(image)` method |
| `src/plantdisease/data/preprocess/leaf_segmentation.py` | `WatershedSegmenter` (primary), `LABSegmenter`, `ColorIndexSegmenter`, `SLICSegmenter` |
| `src/plantdisease/data/preprocess/__init__.py` | Clean exports for all pipeline classes |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/demo_single_image.py` | Demo — processes images, outputs 4×3 visualization grids |
| `scripts/test_pipeline.py` | Full test — 28 images across 7 classes, grids + summary |

---

## Tuning Guide

| Problem | Adjustment |
|---------|-----------|
| Shadows/background still in mask | Increase `texture_weight` (try 6–8) or decrease `shadow_trigger` (try 0.55) |
| Dark/brown tissue being cut away | Decrease `texture_weight` (try 2–3) or increase `dark_boost_threshold` (try 50) |
| Severely wilted/shrivelled leaf | Set `texture_weight=2.0` (relies more on Mahalanobis distance) |
| Too many disease false positives | Increase `disease_threshold` (try 3.0–4.0) |
| Missing subtle disease | Decrease `disease_threshold` (try 2.0) |

---

## Key Design Decisions

1. **Watershed leaf segmentation**: Robust Mahalanobis background model + texture discrimination + adaptive Otsu threshold handles 20+ disease types including dark necrotic tissue, chromatic backgrounds, and cast shadows.

2. **Fused disease detection**: Watershed markers capture distinct colour shifts (blight, spots); Mahalanobis captures subtle deviation from healthy baseline. Union of both maximises recall.

3. **Texture-aware shadow rejection**: Local standard deviation (11×11 window) distinguishes leaf microtexture from smooth shadows. Far more robust than fixed HSV gates.

4. **Disease detection on segmented leaf**: Background is black after segmentation. Both detection methods only consider leaf-mask pixels, eliminating background contamination.

5. **CIELAB colour space**: Both segmentation (Mahalanobis in LAB) and disease detection (Mahalanobis in LAB) use perceptually uniform colour space, robust to illumination variation.

6. **Interior hole filling**: Applied twice in segmentation (pre- and post-Watershed) to ensure disease lesion pits and folded crevices are never excluded from the leaf mask.
