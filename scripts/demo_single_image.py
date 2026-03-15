"""
Demo: Process plant leaf images through the full preprocessing pipeline.

Reads ALL images from an input folder (or a single file), runs all
pipeline stages, and saves a visualization grid for each image into
an output folder.

Usage:
    python scripts/demo_single_image.py --input data/demo_input --output data/demo_output
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plantdisease.data.preprocess.pipeline import PreprocessingPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(input_path: Path) -> list:
    """Return a list of image paths."""
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


def save(img, path, label):
    if img is None:
        print(f"  [SKIP] {label:.<40s} (not available)")
        return
    cv2.imwrite(str(path), img)
    print(f"  [SAVED] {label:.<40s} {path.name}")


def build_summary_grid(result, filename):
    """Build a compact 4-row x 3-col overview grid of the pipeline."""
    cw, ch = 300, 300
    pad, label_h = 8, 22
    panel_h = label_h + ch
    title_h = 36
    text_h = 240
    font = cv2.FONT_HERSHEY_SIMPLEX

    n_rows = 4
    grid_w = 3 * cw + 4 * pad
    grid_h = title_h + n_rows * (panel_h + pad) + pad + text_h

    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    def rs(img):
        if img is None:
            return np.zeros((ch, cw, 3), dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            # RGBA → BGR with white background
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 255.0)
            img = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2BGR) if False else img
            # rembg output is RGB order in the first 3 channels
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR) if False else img
        return cv2.resize(img, (cw, ch))

    def rgba_to_display(rgba):
        """Convert RGBA (RGB order) to BGR for display, with white bg."""
        if rgba is None:
            return np.zeros((ch, cw, 3), dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        blended = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
        return cv2.resize(blended[:, :, ::-1], (cw, ch))  # RGB→BGR

    # Title
    sev = f"{result.severity_percent:.1f}% [{severity_label(result.severity_percent)}]"
    title = f"Pipeline Demo  |  {filename}  |  Severity: {sev}"
    cv2.rectangle(grid, (0, 0), (grid_w, title_h), (255, 255, 255), -1)
    cv2.putText(grid, title, (pad, title_h - 12), font, 0.45,
                (40, 40, 140), 1, cv2.LINE_AA)

    # Prepare visualisation panels
    original_r = cv2.resize(result.original, (cw, ch))
    bg_removed_r = rgba_to_display(result.background_removed)
    resized_r = rs(result.resized)

    # Shadow mask visualisation
    shadow_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
    if result.shadow_mask is not None:
        sm = cv2.resize(result.shadow_mask, (cw, ch))
        shadow_vis[sm > 0] = (0, 0, 200)  # Red for shadow pixels

    shadow_removed_r = rs(result.shadow_removed)

    # Leaf mask visualisation
    leaf_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
    if result.leaf_mask is not None:
        m = cv2.resize(result.leaf_mask, (cw, ch))
        leaf_vis[m > 0] = (0, 200, 0)

    # Disease mask visualisation
    dm_r = cv2.resize(result.disease_mask, (cw, ch)) if result.disease_mask is not None else np.zeros((ch, cw), dtype=np.uint8)
    disease_vis = np.full((ch, cw, 3), 235, dtype=np.uint8)
    disease_vis[dm_r > 0] = (50, 50, 200)

    # Yellow mask visualisation
    yellow_vis = np.full((ch, cw, 3), 235, dtype=np.uint8)
    if result.yellow_mask is not None:
        ym = cv2.resize(result.yellow_mask, (cw, ch))
        yellow_vis[ym > 0] = (0, 255, 255)  # Yellow

    # Brown mask visualisation
    brown_vis = np.full((ch, cw, 3), 235, dtype=np.uint8)
    if result.brown_mask is not None:
        bm = cv2.resize(result.brown_mask, (cw, ch))
        brown_vis[bm > 0] = (30, 80, 180)  # Brown

    overlay_r = rs(result.disease_overlay)

    STEP_BG = (240, 222, 200)
    SEG_BG  = (210, 238, 210)
    DET_BG  = (230, 215, 235)
    EXT_BG  = (210, 218, 245)

    panels = [
        [("STEP 1: Original",               STEP_BG, original_r),
         ("STEP 1: BG Removed (rembg)",      SEG_BG,  bg_removed_r),
         ("STEP 2: Resized 300x300",         STEP_BG, resized_r)],
        [("STEP 3: Shadow Mask",             DET_BG,  shadow_vis),
         ("STEP 3: Shadow Removed",          STEP_BG, shadow_removed_r),
         ("Leaf Mask (from alpha)",          SEG_BG,  leaf_vis)],
        [("STEP 4: Yellow Regions",          DET_BG,  yellow_vis),
         ("STEP 4: Brown Regions",           DET_BG,  brown_vis),
         ("STEP 4: Disease Mask (combined)", DET_BG,  disease_vis)],
        [("Disease Overlay",                 EXT_BG,  overlay_r),
         ("", STEP_BG, np.full((ch, cw, 3), 240, dtype=np.uint8)),
         ("", STEP_BG, np.full((ch, cw, 3), 240, dtype=np.uint8))],
    ]

    for ri, row in enumerate(panels):
        y0 = title_h + pad + ri * (panel_h + pad)
        for ci, (label, bg, img) in enumerate(row):
            x0 = pad + ci * (cw + pad)
            cv2.rectangle(grid, (x0, y0), (x0 + cw, y0 + label_h), bg, -1)
            if label:
                cv2.putText(grid, label, (x0 + 4, y0 + label_h - 7),
                            font, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
            grid[y0 + label_h: y0 + panel_h, x0: x0 + cw] = img

    # Severity text block
    ty0 = title_h + pad + n_rows * (panel_h + pad)
    tx0, tx1 = pad, grid_w - pad
    ty1 = ty0 + text_h - pad

    cv2.rectangle(grid, (tx0, ty0), (tx1, ty1), (255, 255, 255), -1)
    cv2.rectangle(grid, (tx0, ty0), (tx1, ty1), (80, 80, 80), 1)

    sev = severity_label(result.severity_percent)

    lines = [
        (f"SEVERITY ANALYSIS  |  File: {filename}", True),
        ("", False),
        (f"Leaf Area: {result.total_leaf_pixels:,} pixels    |    "
         f"Disease Area: {result.diseased_pixels:,} pixels", False),
        (f"Yellow: {result.yellow_pixels:,} px    |    "
         f"Brown: {result.brown_pixels:,} px", False),
        (f"Severity: {result.severity_percent:.1f}%  [{sev}]", False),
        ("", False),
        ("Pipeline:", True),
        ("STEP 1: Remove Background (rembg deep-learning model)", False),
        ("STEP 2: Resize to 300x300 (Lanczos interpolation)", False),
        ("STEP 3: Shadow Removal (HSV colour-space thresholds)", False),
        ("STEP 4: HSV Disease Segmentation (yellow + brown thresholds)", False),
        ("STEP 5: Severity Calculation (diseased / leaf pixels)", False),
        ("STEP 6: Data Normalisation (pixel values to [0,1])", False),
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


def main():
    parser = argparse.ArgumentParser(
        description="Process leaf images through the full pipeline."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an image file OR a folder.",
    )
    parser.add_argument(
        "--output", "-o", default="data/demo_output",
        help="Output folder (default: data/demo_output).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    img_paths = find_images(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PREPROCESSING PIPELINE DEMO")
    print(f"  Input:  {input_path}")
    print(f"  Images found: {len(img_paths)}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 64)

    pipe = PreprocessingPipeline()

    header = f"{'#':<4} {'Filename':<45} {'Leaf px':>9} {'Severity':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for idx, img_path in enumerate(img_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"{idx:<4} {img_path.name:<45} SKIP (unreadable)")
            continue

        result = pipe.run(image)

        print(f"{idx:<4} {img_path.name:<45} "
              f"{result.total_leaf_pixels:>9,} "
              f"{result.severity_percent:>8.1f}% [{severity_label(result.severity_percent)}]")

        stem = img_path.stem
        grid = build_summary_grid(result, img_path.name)
        grid_path = output_dir / f"{stem}_FULL_GRID.jpg"
        cv2.imwrite(str(grid_path), grid)

    print(f"\n{'='*64}")
    print(f"  Processed {len(img_paths)} image(s)")
    print(f"  All outputs saved to: {output_dir.resolve()}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
