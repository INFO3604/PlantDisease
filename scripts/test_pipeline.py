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


def build_grid(
    original: np.ndarray,
    result: PipelineResult,
    class_name: str,
    image_name: str,
    cell_size: tuple = (256, 256),
) -> np.ndarray:
    """
    Build a 2-row x 3-column grid for one image:

    Row 1:  Original  |  Enhanced (WB+Denoise+AGCWD)  |  Leaf Mask
    Row 2:  Segmented Leaf  |  Disease Mask  |  Disease Overlay (on segmented)
    """
    cw, ch = cell_size
    grid = np.zeros((2 * ch, 3 * cw, 3), dtype=np.uint8)

    short_class = class_name.replace("Tomato___", "")

    # Row 1, Col 0 — Original
    cell = cv2.resize(original, (cw, ch))
    put_text(cell, "Original", (4, 18), scale=0.45)
    put_text(cell, short_class, (4, ch - 10), scale=0.4, color=(180, 255, 180))
    grid[0:ch, 0:cw] = cell

    # Row 1, Col 1 — Contrast-enhanced
    cell = cv2.resize(result.contrast_enhanced, (cw, ch))
    put_text(cell, "Enhanced", (4, 18), scale=0.45)
    grid[0:ch, cw:2*cw] = cell

    # Row 1, Col 2 — Leaf mask
    if result.leaf_mask is not None:
        mask_vis = cv2.cvtColor(cv2.resize(result.leaf_mask, (cw, ch)),
                                cv2.COLOR_GRAY2BGR)
        put_text(mask_vis, f"Leaf Mask {result.mask_ratio:.0%}", (4, 18), scale=0.45)
    else:
        mask_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
        put_text(mask_vis, "Seg FAILED", (4, 18), color=(0, 0, 255))
    grid[0:ch, 2*cw:3*cw] = mask_vis

    # Row 2, Col 0 — Segmented leaf
    if result.segmented_leaf is not None:
        cell = cv2.resize(result.segmented_leaf, (cw, ch))
        put_text(cell, "Segmented Leaf", (4, 18), scale=0.45)
    else:
        cell = np.zeros((ch, cw, 3), dtype=np.uint8)
        put_text(cell, "N/A", (4, 18))
    grid[ch:2*ch, 0:cw] = cell

    # Row 2, Col 1 — Disease mask
    if result.disease_mask is not None:
        dm_vis = cv2.cvtColor(cv2.resize(result.disease_mask, (cw, ch)),
                              cv2.COLOR_GRAY2BGR)
        put_text(dm_vis, f"Disease Mask {result.severity_percent:.1f}%",
                 (4, 18), scale=0.45)
    else:
        dm_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
        put_text(dm_vis, "N/A", (4, 18))
    grid[ch:2*ch, cw:2*cw] = dm_vis

    # Row 2, Col 2 — Disease overlay ON SEGMENTED LEAF (color-coded)
    if result.disease_overlay is not None:
        cell = cv2.resize(result.disease_overlay, (cw, ch))
        put_text(cell, f"Overlay  Sev: {result.severity_percent:.1f}%",
                 (4, 18), scale=0.45)
    else:
        cell = np.zeros((ch, cw, 3), dtype=np.uint8)
        put_text(cell, "N/A", (4, 18))
    grid[ch:2*ch, 2*cw:3*cw] = cell

    return grid


def build_legend(width: int = 768, height: int = 40) -> np.ndarray:
    """Build a colour legend bar for the disease overlay."""
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    items = [
        ("Dark Brown -> Red", (0, 0, 255)),
        ("Light Brown -> Orange", (0, 120, 255)),
        ("Yellow -> Yellow", (0, 255, 255)),
        ("Other -> Magenta", (255, 0, 200)),
    ]

    x = 10
    for label_text, bgr_color in items:
        # Draw colour swatch
        cv2.rectangle(legend, (x, 8), (x + 20, 30), bgr_color, -1)
        cv2.rectangle(legend, (x, 8), (x + 20, 30), (200, 200, 200), 1)
        # Draw label
        cv2.putText(legend, label_text, (x + 25, 25), font, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        x += 180

    return legend


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

        grid = build_grid(image, res, cls_name, img_path.name)
        all_grids.append(grid)

        safe = f"{cls_name}_{img_path.stem}".replace(" ", "_")
        cv2.imwrite(str(OUTPUT_DIR / f"{safe}.jpg"), grid)

    # Combined grid with legend
    if all_grids:
        legend = build_legend(all_grids[0].shape[1])
        sep = np.ones((4, all_grids[0].shape[1], 3), dtype=np.uint8) * 80
        parts = [legend, sep]
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
