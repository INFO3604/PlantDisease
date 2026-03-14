"""
Demo: Process plant leaf images through the full preprocessing pipeline.

Reads ALL images from an input folder (or a single file), runs all
pipeline stages, and saves a visualization grid for each image into
an output folder.

Usage:
    # Process all images in a folder:
    python scripts/demo_single_image.py --input data/demo_input --output data/demo_output

    # Process a specific image file:
    python scripts/demo_single_image.py --input path/to/leaf.jpg --output data/demo_output
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plantdisease.data.preprocess.pipeline import PreprocessingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(input_path: Path) -> list:
    """Return a list of image paths — either a single file or all images
    found inside a directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        imgs = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not imgs:
            raise FileNotFoundError(
                f"No image files found in '{input_path}'. "
                f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}"
            )
        return imgs
    raise FileNotFoundError(f"Path does not exist: {input_path}")


def severity_label(pct: float) -> str:
    if pct < 5:
        return "Healthy / Trace"
    elif pct < 15:
        return "Low"
    elif pct < 30:
        return "Moderate"
    elif pct < 50:
        return "Severe"
    return "Very Severe"


def save(img: np.ndarray, path: Path, label: str):
    """Save an image and print confirmation."""
    if img is None:
        print(f"  [SKIP] {label:.<40s} (not available)")
        return
    cv2.imwrite(str(path), img)
    print(f"  [SAVED] {label:.<40s} {path.name}")


# ---------------------------------------------------------------------------
# Build summary grid (same 4x3 layout as test_pipeline.py)
# ---------------------------------------------------------------------------

def build_summary_grid(result, original_resized, filename):
    """Build a compact 4-row x 3-col overview grid of the full pipeline."""
    cw, ch = 256, 256
    pad, label_h = 8, 22
    panel_h = label_h + ch
    title_h = 36
    text_h = 200
    font = cv2.FONT_HERSHEY_SIMPLEX

    grid_w = 3 * cw + 4 * pad
    grid_h = title_h + 4 * (panel_h + pad) + pad + text_h

    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    def rs(img):
        if img is None:
            return np.zeros((ch, cw, 3), dtype=np.uint8)
        return cv2.resize(img, (cw, ch))

    # Title
    sev = f"{result.severity_percent:.1f}% [{severity_label(result.severity_percent)}]"
    title = f"Pipeline Demo  |  {filename}  |  Severity: {sev}"
    cv2.rectangle(grid, (0, 0), (grid_w, title_h), (255, 255, 255), -1)
    cv2.putText(grid, title, (pad, title_h - 12), font, 0.45,
                (40, 40, 140), 1, cv2.LINE_AA)

    # Disease mask visual (red on light bg)
    dm_r = cv2.resize(result.disease_mask, (cw, ch)) if result.disease_mask is not None else np.zeros((ch, cw), dtype=np.uint8)
    disease_vis = np.full((ch, cw, 3), 235, dtype=np.uint8)
    disease_vis[dm_r > 0] = (50, 50, 200)

    # Leaf mask visual (green on black)
    leaf_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
    if result.leaf_mask is not None:
        m = cv2.resize(result.leaf_mask, (cw, ch))
        leaf_vis[m > 0] = (0, 200, 0)

    # Combined colour
    if result.segmented_leaf is not None and result.leaf_mask is not None:
        seg_r = rs(result.segmented_leaf)
        mask_r = cv2.resize(result.leaf_mask, (cw, ch))
        combined = seg_r.copy().astype(np.float64)
        healthy = (mask_r > 0) & (dm_r == 0)
        if np.any(healthy):
            combined[healthy] = 0.7 * combined[healthy] + 0.3 * np.array([0, 180, 0])
        if np.any(dm_r > 0):
            combined[dm_r > 0] = 0.5 * combined[dm_r > 0] + 0.5 * np.array([0, 0, 220])
        combined = np.clip(combined, 0, 255).astype(np.uint8)
    else:
        combined = np.zeros((ch, cw, 3), dtype=np.uint8)

    # Grayscale
    if result.segmented_leaf is not None:
        gray = cv2.cvtColor(rs(result.segmented_leaf), cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray_bgr = np.zeros((ch, cw, 3), dtype=np.uint8)

    # Disease on leaf (contours)
    if result.disease_mask is not None and result.segmented_leaf is not None:
        seg_r = rs(result.segmented_leaf)
        extract_dm = seg_r.copy()
        contours, _ = cv2.findContours(dm_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(extract_dm, contours, -1, (0, 0, 255), 2)
        dm_bool = dm_r > 0
        extract_dm[dm_bool] = (0.4 * extract_dm[dm_bool].astype(np.float64) + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    else:
        extract_dm = np.zeros((ch, cw, 3), dtype=np.uint8)

    # Disease only (on black)
    if result.disease_mask is not None and result.segmented_leaf is not None:
        seg_r = rs(result.segmented_leaf)
        extract_do = np.zeros((ch, cw, 3), dtype=np.uint8)
        extract_do[dm_r > 0] = seg_r[dm_r > 0]
    else:
        extract_do = np.zeros((ch, cw, 3), dtype=np.uint8)

    # Label colours (BGR)
    STEP_BG   = (240, 222, 200)
    SEG_BG    = (210, 238, 210)
    DET_BG    = (230, 215, 235)
    COMB_BG   = (205, 238, 248)
    EXT_BG    = (210, 218, 245)

    panels = [
        [("STEP 0: Original",           STEP_BG, rs(original_resized)),
         ("STEP 1: White Balanced",      STEP_BG, rs(result.white_balanced)),
         ("STEP 2: Denoised",            STEP_BG, rs(result.denoised))],
        [("STEP 3: Contrast (AGCWD)",    STEP_BG, rs(result.contrast_enhanced)),
<<<<<<< HEAD
         ("STEP 5: SAM Leaf Mask",       SEG_BG,  leaf_vis),
         ("STEP 6: Masked AGCWD Leaf",   SEG_BG,  rs(result.segmented_leaf))],
        [("STEP 8: Disease Mask",        DET_BG,  disease_vis),
=======
         ("STEP 4: Leaf Mask",           SEG_BG,  leaf_vis),
         ("STEP 4: Isolated Leaf",       SEG_BG,  rs(result.segmented_leaf))],
        [("STEP 5: Disease Mask",        DET_BG,  disease_vis),
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
         ("COMBINED: Colour",            COMB_BG, combined),
         ("COMBINED: Greyscale",         COMB_BG, gray_bgr)],
        [("EXTRACT: Disease on Leaf",    EXT_BG,  extract_dm),
         ("EXTRACT: Disease Only",       EXT_BG,  extract_do),
         ("EXTRACT: Overlay",            EXT_BG,  rs(result.disease_overlay))],
    ]

    for ri, row in enumerate(panels):
        y0 = title_h + pad + ri * (panel_h + pad)
        for ci, (label, bg, img) in enumerate(row):
            x0 = pad + ci * (cw + pad)
            cv2.rectangle(grid, (x0, y0), (x0 + cw, y0 + label_h), bg, -1)
            cv2.putText(grid, label, (x0 + 4, y0 + label_h - 7),
                        font, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
            grid[y0 + label_h: y0 + panel_h, x0: x0 + cw] = img

    # ===================================================================
    # SEVERITY ANALYSIS TEXT BLOCK
    # ===================================================================
    ty0 = title_h + pad + 4 * (panel_h + pad)
    tx0, tx1 = pad, grid_w - pad
    ty1 = ty0 + text_h - pad

    cv2.rectangle(grid, (tx0, ty0), (tx1, ty1), (255, 255, 255), -1)
    cv2.rectangle(grid, (tx0, ty0), (tx1, ty1), (80, 80, 80), 1)

    sev = severity_label(result.severity_percent)

    lines = [
        (f"SEVERITY ANALYSIS  |  File: {filename}", True),
        ("", False),
        (f"Main Leaf Area: {result.total_leaf_pixels:,} pixels    |    "
         f"Disease Area: {result.diseased_pixels:,} pixels", False),
        (f"Severity: {result.severity_percent:.1f}%  [{sev}]", False),
        ("", False),
        ("Complete Processing Pipeline:", True),
        ("STEP 0: Input Image (Lanczos resize to 256x256)", False),
        ("STEP 1: Gray-World White Balance - Correct colour casts", False),
        ("STEP 2: Bilateral Filter (d=9) - Edge-preserving denoising", False),
        ("STEP 3: AGCWD - Adaptive Gamma Correction with Weighting Distribution",
         False),
<<<<<<< HEAD
        ("STEP 5: SAM Full Leaf Isolation - Binary leaf mask", False),
        ("STEP 6: Apply SAM Mask to AGCWD - Zero background", False),
        ("STEP 7: Convert masked leaf to HSV/LAB", False),
        ("STEP 8: Watershed on Hue/a* inside SAM mask only", False),
=======
        ("STEP 4: LAB Chroma Segmentation - Isolate leaf "
         "(green + brown tissue)", False),
        ("STEP 5: Mahalanobis Distance (threshold=2.5) - "
         "Disease detection on segmented leaf", False),
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
        ("EXTRACTION: Color-coded overlay "
         "(Brown->Red, Yellow->Yellow, Other->Crimson)", False),
    ]

    ly = ty0 + 16
    for line_text, is_header in lines:
        if not line_text:
            ly += 14
            continue
        sc = 0.40 if is_header else 0.36
        if is_header and ly < ty0 + 20:
            (tw2, _), _ = cv2.getTextSize(line_text, font, sc, 1)
            lx = max(tx0 + 10, (grid_w - tw2) // 2)
        else:
            lx = tx0 + 20
        cv2.putText(grid, line_text, (lx, ly), font, sc,
                    (0, 0, 0), 1, cv2.LINE_AA)
        ly += 14

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process leaf images through the full pipeline."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an image file OR a folder (all images will be processed).",
    )
    parser.add_argument(
        "--output", "-o", default="data/demo_output",
        help="Output folder for the results (default: data/demo_output).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Resolve images
    img_paths = find_images(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PREPROCESSING PIPELINE DEMO")
    print(f"  Input:  {input_path}")
    print(f"  Images found: {len(img_paths)}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 64)

    # Build pipeline
    pipe = PreprocessingPipeline(
        segmentation_method="lab_astar",
        disease_threshold=2.5,
    )

    header = f"{'#':<4} {'Filename':<45} {'Seg OK':<8} {'Mask%':>7} {'Severity':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for idx, img_path in enumerate(img_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"{idx:<4} {img_path.name:<45} SKIP (unreadable)")
            continue

        result = pipe.run(image)

        ok = "YES" if result.segmentation_success else "NO"
        print(f"{idx:<4} {img_path.name:<45} {ok:<8} "
              f"{result.mask_ratio:>6.1%} "
              f"{result.severity_percent:>8.1f}% [{severity_label(result.severity_percent)}]")

        # Save grid
        stem = img_path.stem
        grid = build_summary_grid(result, result.resized, img_path.name)
        grid_path = output_dir / f"{stem}_FULL_GRID.jpg"
        cv2.imwrite(str(grid_path), grid)

    print(f"\n{'='*64}")
    print(f"  Processed {len(img_paths)} image(s)")
    print(f"  All outputs saved to: {output_dir.resolve()}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
