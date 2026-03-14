<<<<<<< HEAD
# Preprocessing Pipeline

This document describes the preprocessing pipeline used by the project and how SAM (Segment Anything Model) is integrated.

Pipeline overview (ordered):

1. Resize (Lanczos) — resize to target size (default 256×256).
2. White-balance (Gray-World) — correct colour casts.
3. Denoise (Bilateral) — edge-preserving smoothing.
4. Contrast enhancement (AGCWD) — adaptive gamma correction on luminance.
5. SAM — Full Leaf Isolation
   - SAM runs on the AGCWD image.
   - Output: binary leaf mask (255 = leaf).
6. Apply SAM Mask to AGCWD image
   - Zero out background pixels; only leaf pixels remain.
7. Convert masked leaf → HSV / LAB
   - White-balance makes hue channel reliable for disease cues.
8. Watershed — Disease Region Segmentation
   - Watershed runs ONLY inside the SAM mask.
   - Seed markers are derived from Hue (HSV) and a\* (LAB).

## SAM details and checkpoint

- The pipeline expects a SAM checkpoint at `models/sam_vit_b.pth` inside the repository.
- You can override this with the `SAM_CHECKPOINT` environment variable.
- On CPU-only machines a lower-cost SAM configuration is used (smaller point sampling, smaller input size). For best performance, run with CUDA-enabled PyTorch.

## Running the demo

Process all demo images and save visualizations to an output folder:

```bash
c:/Users/robyn/PlantDisease/.venv/Scripts/python.exe scripts/demo_single_image.py --input data/demo_input --output data/test_output
```

## Quick troubleshooting

- If the pipeline reports "SAM checkpoint not found", ensure the file exists at `models/sam_vit_b.pth` or set `SAM_CHECKPOINT`.
- If SAM inference is slow on CPU, consider installing CUDA drivers and running on GPU or reducing `points_per_side` via code.

## Contact

For questions about preprocessing or to adjust segmentation weights, see `src/plantdisease/data/preprocess/leaf_segmentation.py` and `src/plantdisease/data/preprocess/sam_segmenation.py`.
=======
# Preprocessing Pipeline Documentation

## Overview

This document describes the **image preprocessing pipeline** for automated plant disease detection on Solanaceae (tomato) leaf images from the PlantVillage dataset. The pipeline uses **triple-channel CIELAB segmentation** (no GrabCut) with **Mahalanobis-distance disease detection**, **shadow rejection**, and **colour-coded disease visualisation**.

All disease detection runs on the **segmented leaf** (background removed) to eliminate shadow false-positives.

---

## Why This Approach?

| Concern | Supervisor's Pipeline | Our Pipeline |
|---------|----------------------|--------------|
| **Segmentation** | GrabCut (already implemented) | Triple-channel CIELAB (completely independent method) |
| **Shadow handling** | Disease overlay on full image - shadows misclassified | Disease overlay on **segmented leaf** (black bg) + explicit shadow rejection |
| **Brown/dried tissue** | May miss necrotic regions | Triple mask (a\* + chroma + b\*) + interior hole filling captures all tissue |
| **Dependencies** | GrabCut initialisation, slow | Pure OpenCV, fast, no manual tuning |
| **Disease categories** | Single colour | 4-category colour-coded overlay (Red, Orange, Yellow, Crimson) |

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
  |      Edge-preserving smoothing - keeps disease boundaries sharp
  |      while removing sensor noise from flat regions.
  |
  +-- 4. Contrast Enhancement (AGCWD)
  |      Adaptive Gamma Correction with Weighting Distribution.
  |      Per-intensity adaptive gamma based on luminance PDF/CDF.
  |      Dark images brightened more; bright images left alone.
  |
  +-- 5. Leaf Segmentation (Triple-Channel CIELAB)
  |      Mask 1: Inverted a* + Otsu -> captures green tissue
  |      Mask 2: Chroma sqrt(a*^2 + b*^2) + Otsu -> captures colourful brown/red
  |      Mask 3: b* channel + Otsu -> captures dried/tan tissue (warm b*)
  |      Union of all three masks -> morphological cleanup -> largest component
  |      Interior holes filled so dried inner tissue is never excluded.
  |      -> Binary leaf mask + segmented leaf (background = black)
  |
  +-- 6. Disease Detection (Mahalanobis distance, threshold=2.5)
  |      Runs on SEGMENTED LEAF only (no background contamination).
  |      Healthy green pixels define reference distribution (mean + covariance).
  |      Pixels with Mahalanobis distance > 2.5 classified as diseased.
  |      Shadow rejection: removes low-saturation dark pixels (S<70 & V<85)
  |      and very dark pixels (V<40).
  |      Morphological cleanup + re-apply leaf mask to prevent overflow.
  |      -> Disease mask + severity percentage
  |
  +-- 7. Colour-Coded Disease Overlay (alpha=0.75)
         4 categories based on HSV colour of diseased pixels:
           Dark brown / necrotic -> Deep Red
           Light brown / tan    -> Bright Orange
           Yellow / chlorotic   -> Bright Yellow
           Other diseased       -> Dark Crimson
         White contour outlines drawn around each diseased region.
```

---

## Segmentation: Triple-Channel CIELAB

Three masks are computed and combined with OR to capture **all** leaf tissue:

| Mask | Channel | What It Captures | Why Needed |
|------|---------|------------------|------------|
| **Mask 1** | Inverted a\* | Green tissue (healthy leaves) | a\* encodes green-red axis; green = low a\* |
| **Mask 2** | Chroma sqrt(a\*^2 + b\*^2) | Colourful brown/red tissue | High chroma despite positive a\* |
| **Mask 3** | b\* channel | Dried/tan/warm tissue | Desaturated brown has warm b\* but low chroma |

After combining, **interior holes are filled** by finding the external contour and filling it solid. This ensures dried patches enclosed within the leaf boundary are never excluded.

### Method Comparison (initial 14-image evaluation)

| Method | Success Rate | Dependencies | Speed | Notes |
|--------|-------------|--------------|-------|-------|
| ExG Color-Index | 13/14 (93%) | OpenCV only | Fast | Failed on Target Spot (mask 4.2%) |
| **LAB a\*-channel** | **14/14 (100%)** | **OpenCV only** | **Fast** | **Selected as primary method** |
| SLIC Superpixel | 14/14 (100%) | scikit-image | Slow | Over-detection tendency |

LAB a\*-channel was selected and later enhanced to triple-channel (a\* + chroma + b\*) with hole filling.

---

## Shadow Rejection

Shadows on leaf surfaces produce dark, desaturated regions that mimic necrotic tissue. After Mahalanobis thresholding, a shadow rejection step removes false positives:

- **Low-saturation dark pixels**: S < 70 AND V < 85 -> classified as shadow, removed
- **Very dark pixels**: V < 40 -> unconditionally removed (deep shadows / background remnants)

This preserves genuine disease (which retains moderate-to-high saturation even when dark) while eliminating shadow artefacts.

---

## Disease Overlay Colours

| Category | HSV Criteria | Overlay Colour (BGR) | Clinical Meaning |
|----------|-------------|---------------------|-----------------|
| Dark brown / necrotic | H<22 & V<=80; or H<15 & S<=50 | Deep Red (30,30,255) | Dead tissue, advanced necrosis |
| Light brown / tan | H in [8,22), S>25, V>80 | Bright Orange (0,140,255) | Early browning, lesion margins |
| Yellow / chlorotic | H in [15,35], S>40 | Bright Yellow (0,255,255) | Chlorosis, nutrient deficiency |
| Other diseased | Does not match above | Dark Crimson (30,0,180) | Atypical discolouration |

Alpha blending: 0.75 (bold overlay + original texture visible). White contour outlines around each diseased region.

---

## Test Results

Tested on **28 PlantVillage images** (4 per class across 7 tomato disease classes). All disease detection on segmented leaf with shadow rejection enabled.

| # | Disease Class | Seg OK | Leaf Mask % | Severity % |
|---|--------------|--------|-------------|------------|
| 1 | Bacterial spot | YES | 45.1% | 13.1% |
| 2 | Bacterial spot | YES | 48.8% | 16.3% |
| 3 | Bacterial spot | YES | 43.0% | 8.6% |
| 4 | Bacterial spot | YES | 52.6% | 19.6% |
| 5 | Early blight | YES | 69.9% | 48.9% |
| 6 | Early blight | YES | 36.7% | 8.9% |
| 7 | Early blight | YES | 30.2% | 14.7% |
| 8 | Early blight | YES | 75.4% | 33.0% |
| 9 | Late blight | YES | 22.5% | 5.7% |
| 10 | Late blight | YES | 55.8% | 36.3% |
| 11 | Late blight | YES | 22.3% | 7.4% |
| 12 | Late blight | YES | 34.3% | 12.3% |
| 13 | Leaf Mold | YES | 42.0% | 10.4% |
| 14 | Leaf Mold | YES | 31.5% | 10.8% |
| 15 | Leaf Mold | YES | 41.8% | 9.3% |
| 16 | Leaf Mold | YES | 28.1% | 3.1% |
| 17 | Septoria leaf spot | YES | 19.7% | 13.2% |
| 18 | Septoria leaf spot | YES | 41.2% | 8.4% |
| 19 | Septoria leaf spot | YES | 54.8% | 11.5% |
| 20 | Septoria leaf spot | YES | 44.5% | 8.6% |
| 21 | Spider mites | YES | 34.7% | 10.4% |
| 22 | Spider mites | YES | 47.2% | 15.3% |
| 23 | Spider mites | YES | 31.4% | 4.3% |
| 24 | Spider mites | YES | 28.5% | 6.9% |
| 25 | Target Spot | YES | 43.9% | 11.1% |
| 26 | Target Spot | YES | 39.5% | 21.3% |
| 27 | Target Spot | YES | 48.3% | 5.1% |
| 28 | Target Spot | YES | 48.6% | 9.6% |

**28/28 images segmented successfully (100%)**

Severity ranges are clinically plausible: 3.1% (mild Leaf Mold) to 48.9% (severe Early Blight). No shadow false-positives observed.

---

## Visualization Output

The test script (`scripts/test_pipeline.py`) produces a **4x3 comprehensive grid** for each image:

| | Column 1 | Column 2 | Column 3 |
|---|----------|----------|----------|
| **Row 1** | Original Image | White Balanced | Denoised |
| **Row 2** | Contrast Enhanced | Leaf Mask (green on black) | Isolated Leaf |
| **Row 3** | Disease Mask (red on light bg) | Combined Colour | Combined Greyscale |
| **Row 4** | Disease on Leaf (contours) | Disease Only (on black) | Colour-Coded Overlay |

Each grid includes a title bar and a severity analysis text block at the bottom showing pixel counts, severity percentage, and the full pipeline step list.

Output location: `data/preprocessed_output/pipeline_test/`

---

## Scripts & Commands

### Demo Script (recommended for presentations)

Process all images in an input folder and output a visualization grid per image:

```bash
# Put leaf images into data/demo_input/, then run:
python scripts/demo_single_image.py --input data/demo_input --output data/demo_output

# Or process a single specific image:
python scripts/demo_single_image.py --input path/to/leaf.jpg --output data/demo_output
```

Each image produces a `_FULL_GRID.jpg` file in the output folder with the 4x3 pipeline visualization and severity analysis.

### Full Test Suite (28 images, 7 disease classes)

```bash
python scripts/test_pipeline.py
```

Produces per-image grids + a combined summary grid. Prints a results table to the console.

### Python API

```python
import cv2
from plantdisease.data.preprocess import PreprocessingPipeline

# Create pipeline (LAB a* is the default segmentation method)
pipe = PreprocessingPipeline()

# Run on an image
image = cv2.imread("path/to/leaf.jpg")
result = pipe.run(image)

# Access outputs
print(f"Segmentation: {result.segmentation_method}")
print(f"Success: {result.segmentation_success}")
print(f"Leaf mask ratio: {result.mask_ratio:.1%}")
print(f"Disease severity: {result.severity_percent:.1f}%")
print(f"Diseased pixels: {result.diseased_pixels:,}")
print(f"Total leaf pixels: {result.total_leaf_pixels:,}")
print(f"Steps applied: {result.steps_applied}")

# Save outputs
cv2.imwrite("segmented.jpg", result.segmented_leaf)
cv2.imwrite("disease_overlay.jpg", result.disease_overlay)
cv2.imwrite("leaf_mask.png", result.leaf_mask)
cv2.imwrite("disease_mask.png", result.disease_mask)
```

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

## Project Files

### Pipeline Source Code

| File | Purpose |
|------|---------|
| `src/plantdisease/data/preprocess/pipeline.py` | `PreprocessingPipeline` class - full end-to-end pipeline with `run(image)` method |
| `src/plantdisease/data/preprocess/leaf_segmentation.py` | Three segmenter classes: `LABSegmenter` (primary), `ColorIndexSegmenter`, `SLICSegmenter` |
| `src/plantdisease/data/preprocess/__init__.py` | Clean exports for all pipeline classes |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/demo_single_image.py` | Demo script - processes all images in input folder, outputs visualization grids |
| `scripts/test_pipeline.py` | Full test script - 28 images across 7 classes, comprehensive grids + combined summary |

### Documentation

| File | Purpose |
|------|---------|
| `PREPROCESSING_README.md` | This document |
| `Methodology_Preprocessing_Pipeline.docx` | Formal methodology write-up for report submission |

---

## Key Design Decisions

1. **Disease detection on segmented leaf**: Background is black after segmentation. Mahalanobis computation only considers leaf-mask pixels, so shadows, table edges, and labels cannot be misclassified as disease.

2. **Triple-channel CIELAB segmentation**: Single a\* channel misses extremely dried/brown tissue. Adding chroma and b\* masks captures the full spectrum of leaf tissue (green, brown, dried, necrotic). Interior hole filling ensures no enclosed tissue is excluded.

3. **Shadow rejection post-processing**: Even after segmentation, leaf-surface shadows can trigger false positives. Explicit HSV-based shadow filtering (low S + low V) removes these while preserving genuine disease.

4. **CIELAB colour space throughout**: Both segmentation (a\*/chroma/b\*) and disease detection (Mahalanobis in LAB) use CIELAB, which is perceptually uniform and separates luminance from chrominance, making both steps robust to lighting variation.

5. **No GrabCut**: The supervisor's pipeline uses GrabCut. This pipeline is completely independent - different algorithm, different theory, different implementation.

6. **Morphological safety**: Both segmentation and disease detection include open/close operations, largest-component extraction, and leaf-mask re-application after morphological closing to prevent mask overflow beyond the leaf boundary.
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
