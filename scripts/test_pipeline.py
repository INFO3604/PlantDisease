"""
Test the preprocessing pipeline (LAB a*-channel segmentation).

Runs the pipeline on sample PlantVillage images:
  - Disease detection is done on the SEGMENTED leaf (no shadows)
  - Color-coded disease overlay (brown→red, yellow→yellow, etc.)
  - Saves per-image 6-panel grids showing the full pipeline
  - Saves a combined summary grid
  - Prints a console results table

Usage:
    python scripts/test_pipeline.py
"""

import sys
import random
from pathlib import Path

import cv2
import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plantdisease.data.preprocess.pipeline import (
    PreprocessingPipeline,
    PipelineResult,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCHIVE_BASE = Path(
    r"C:\Users\User\Downloads\ABDM\Compressed\archive"
    r"\New Plant Diseases Dataset(Augmented)"
    r"\New Plant Diseases Dataset(Augmented)\train"
)

TEST_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
]

IMAGES_PER_CLASS = 4  # 4 × 7 = 28 images
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed_output" / "pipeline_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_test_images() -> list:
    """Gather sample images from the archive."""
    samples = []
    for cls_name in TEST_CLASSES:
        cls_dir = ARCHIVE_BASE / cls_name
        if not cls_dir.exists():
            print(f"  [SKIP] Class directory not found: {cls_dir}")
            continue
        imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.JPG"))
        if not imgs:
            imgs = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.PNG"))
        if imgs:
            random.seed(42)
            chosen = random.sample(imgs, min(IMAGES_PER_CLASS, len(imgs)))
            for p in chosen:
                samples.append((cls_name, p))
    return samples


def put_text(img, text, pos=(10, 25), scale=0.5, color=(255, 255, 255),
             bg_color=(0, 0, 0), thickness=1):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 2),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def severity_label(pct: float) -> str:
    """Return human-readable severity category."""
    if pct < 5:
        return "Healthy/Trace"
    elif pct < 15:
        return "Low"
    elif pct < 30:
        return "Moderate"
    elif pct < 50:
        return "Severe"
    return "Very Severe"


def build_comprehensive_grid(
    original: np.ndarray,
    result: PipelineResult,
    class_name: str,
    filename: str,
    cell_size: tuple = (256, 256),
) -> np.ndarray:
    """
    Build a 4-row × 3-column comprehensive pipeline visualization.

    Matches the supervisor's multi-panel layout:

    Row 1: Preprocessing   (Original → White Balance → Denoise)
    Row 2: Enhancement     (AGCWD → Leaf Mask → Isolated Leaf)
    Row 3: Detection       (Disease Mask → Color Combined → Grayscale)
    Row 4: Extraction      (Disease Mask on leaf → Disease Only → Overlay)
    + Title bar at the top
    + Severity analysis text block at the bottom
    """
    cw, ch = cell_size
    pad = 10
    label_h = 26
    panel_h = label_h + ch

    title_h = 40
    text_h = 200

    grid_w = 3 * cw + 4 * pad
    grid_h = title_h + 4 * (panel_h + pad) + pad + text_h

    # Light gray background
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    disease_short = class_name.replace("Tomato___", "").replace("_", " ")

    # --- Helper: resize to cell ---
    def rs(img):
        if img is None:
            return np.zeros((ch, cw, 3), dtype=np.uint8)
        return cv2.resize(img, (cw, ch))

    # ===================================================================
    # TITLE BAR
    # ===================================================================
    title_text = f"Complete Pipeline Visualization -- {disease_short}: {filename}"
    cv2.rectangle(grid, (0, 0), (grid_w, title_h), (255, 255, 255), -1)
    (tw, th_), _ = cv2.getTextSize(title_text, font, 0.47, 1)
    tx = max(pad, (grid_w - tw) // 2)
    cv2.putText(grid, title_text, (tx, title_h // 2 + th_ // 2),
                font, 0.47, (40, 40, 140), 1, cv2.LINE_AA)

    # ===================================================================
    # PREPARE SPECIAL VISUALISATION PANELS
    # ===================================================================

    # Resized disease mask (used by several panels below)
    if result.disease_mask is not None:
        dm_r = cv2.resize(result.disease_mask, (cw, ch))
    else:
        dm_r = np.zeros((ch, cw), dtype=np.uint8)

    # -- Leaf mask: green on black (supervisor style) ---
    leaf_mask_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
    if result.leaf_mask is not None:
        m = cv2.resize(result.leaf_mask, (cw, ch))
        leaf_mask_vis[m > 0] = (0, 200, 0)

    # -- Disease mask: red spots on light background ---
    if result.disease_mask is not None:
        disease_mask_vis = np.full((ch, cw, 3), 235, dtype=np.uint8)
        disease_mask_vis[dm_r > 0] = (50, 50, 200)
    else:
        disease_mask_vis = np.zeros((ch, cw, 3), dtype=np.uint8)

    # -- Combined Color (healthy=green tint, diseased=red tint) ---
    if result.segmented_leaf is not None and result.leaf_mask is not None:
        seg_r = rs(result.segmented_leaf)
        mask_r = cv2.resize(result.leaf_mask, (cw, ch))
        combined_color = seg_r.copy().astype(np.float64)
        healthy_m = (mask_r > 0) & (dm_r == 0)
        if np.any(healthy_m):
            combined_color[healthy_m] = (
                0.7 * combined_color[healthy_m]
                + 0.3 * np.array([0, 180, 0])
            )
        if np.any(dm_r > 0):
            combined_color[dm_r > 0] = (
                0.5 * combined_color[dm_r > 0]
                + 0.5 * np.array([0, 0, 220])
            )
        combined_color = np.clip(combined_color, 0, 255).astype(np.uint8)
    else:
        combined_color = np.zeros((ch, cw, 3), dtype=np.uint8)

    # -- Combined Grayscale ---
    if result.segmented_leaf is not None:
        gray = cv2.cvtColor(rs(result.segmented_leaf), cv2.COLOR_BGR2GRAY)
        combined_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        combined_gray = np.zeros((ch, cw, 3), dtype=np.uint8)

    # -- Extract: Disease Mask drawn on leaf (red contours + fill) ---
    if result.disease_mask is not None and result.segmented_leaf is not None:
        seg_r = rs(result.segmented_leaf)
        extract_dm = seg_r.copy()
        contours, _ = cv2.findContours(
            dm_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(extract_dm, contours, -1, (0, 0, 255), 2)
        dm_bool = dm_r > 0
        extract_dm[dm_bool] = (
            0.4 * extract_dm[dm_bool].astype(np.float64)
            + 0.6 * np.array([0, 0, 255])
        ).astype(np.uint8)
    else:
        extract_dm = np.zeros((ch, cw, 3), dtype=np.uint8)

    # -- Extract: Disease Only (diseased pixels on black) ---
    if result.disease_mask is not None and result.segmented_leaf is not None:
        seg_r = rs(result.segmented_leaf)
        extract_do = np.zeros((ch, cw, 3), dtype=np.uint8)
        extract_do[dm_r > 0] = seg_r[dm_r > 0]
    else:
        extract_do = np.zeros((ch, cw, 3), dtype=np.uint8)

    # ===================================================================
    # LABEL BACKGROUND COLOURS (BGR)
    # ===================================================================
    STEP_BG   = (240, 222, 200)   # light blue
    SEG_BG    = (210, 238, 210)   # light green
    DETECT_BG = (230, 215, 235)   # light pink
    COMB_BG   = (205, 238, 248)   # light cream
    EXTR_BG   = (210, 218, 245)   # light salmon

    # ===================================================================
    # PANEL LAYOUT  (4 rows × 3 cols)
    # ===================================================================
    panels = [
        [   # Row 1 — Preprocessing
            ("STEP 0: Original Image",        STEP_BG, rs(original)),
            ("STEP 1: White Balanced",         STEP_BG, rs(result.white_balanced)),
            ("STEP 2: Denoised",               STEP_BG, rs(result.denoised)),
        ],
        [   # Row 2 — Enhancement & Segmentation
            ("STEP 3: Contrast Enhanced",      STEP_BG, rs(result.contrast_enhanced)),
            ("STEP 4: Leaf Mask (LAB Chroma)", SEG_BG,  leaf_mask_vis),
            ("STEP 4: Isolated Leaf",          SEG_BG,  rs(result.segmented_leaf)),
        ],
        [   # Row 3 — Disease detection & combined views
            ("STEP 5: Disease Mask",           DETECT_BG, disease_mask_vis),
            ("COMBINED: Color Output",         COMB_BG,   combined_color),
            ("COMBINED: Grayscale Output",     COMB_BG,   combined_gray),
        ],
        [   # Row 4 — Extraction
            ("EXTRACT: Disease Mask",          EXTR_BG, extract_dm),
            ("EXTRACT: Disease Only",          EXTR_BG, extract_do),
            ("EXTRACT: Overlay on Leaf",       EXTR_BG, rs(result.disease_overlay)),
        ],
    ]

    for row_i, row in enumerate(panels):
        y0 = title_h + pad + row_i * (panel_h + pad)
        for col_i, (label, bg, img) in enumerate(row):
            x0 = pad + col_i * (cw + pad)
            # Label bar
            cv2.rectangle(grid, (x0, y0), (x0 + cw, y0 + label_h), bg, -1)
            cv2.putText(grid, label, (x0 + 5, y0 + label_h - 8),
                        font, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
            # Image
            grid[y0 + label_h : y0 + panel_h, x0 : x0 + cw] = img

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
        (f"SEVERITY ANALYSIS  |  Disease: {disease_short}"
         f"  |  File: {filename}", True),
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
        ("STEP 4: LAB Chroma Segmentation - Isolate leaf "
         "(green + brown tissue)", False),
        ("STEP 5: Mahalanobis Distance (threshold=2.5) - "
         "Disease detection on segmented leaf", False),
        ("EXTRACTION: Color-coded overlay "
         "(Brown->Red, Yellow->Yellow, Other->Crimson)", False),
    ]

    ly = ty0 + 16
    for line_text, is_header in lines:
        if not line_text:
            ly += 14
            continue
        sc = 0.40 if is_header else 0.36
        # Centre the first header line
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
    print("=" * 72)
    print("  PREPROCESSING PIPELINE TEST  —  LAB a*-channel  (no GrabCut)")
    print("  Color-coded disease overlay on SEGMENTED leaf (shadow-free)")
    print("=" * 72)

    samples = collect_test_images()
    if not samples:
        print("ERROR: No test images found. Check ARCHIVE_BASE path.")
        return

    print(f"\nFound {len(samples)} test images across {len(TEST_CLASSES)} classes.\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build pipeline — LAB a* (best method)
    pipe = PreprocessingPipeline(
        segmentation_method="lab_astar",
        disease_threshold=2.5,
    )

    all_grids = []
    successes = 0
    header = (f"{'#':<3} {'Disease Class':<35} {'Seg OK':<8} "
              f"{'Mask%':>7} {'Severity%':>10}")
    print(header)
    print("-" * len(header))

    for i, (cls_name, img_path) in enumerate(samples, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue

        res = pipe.run(image)

        short = cls_name.replace("Tomato___", "").replace("_", " ")
        ok = "YES" if res.segmentation_success else "NO"
        if res.segmentation_success:
            successes += 1
        print(f"{i:<3} {short:<35} {ok:<8} {res.mask_ratio:>6.1%} "
              f"{res.severity_percent:>9.1f}%")

        grid = build_comprehensive_grid(image, res, cls_name, img_path.name)
        all_grids.append(grid)

        safe = f"{cls_name}_{img_path.stem}".replace(" ", "_")
        cv2.imwrite(str(OUTPUT_DIR / f"{safe}.jpg"), grid)

    # Combined grid
    if all_grids:
        sep = np.ones((6, all_grids[0].shape[1], 3), dtype=np.uint8) * 120
        parts = []
        for g in all_grids:
            parts.append(g)
            parts.append(sep)
        combined = np.vstack(parts[:-1])
        combined_path = OUTPUT_DIR / "combined_results.jpg"
        cv2.imwrite(str(combined_path), combined)
        print(f"\nCombined grid saved -> {combined_path}")

    total = len(samples)
    print(f"\n{successes}/{total} images segmented successfully "
          f"({successes/total*100:.0f}%)")
    print(f"All outputs saved to:  {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
