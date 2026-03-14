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
