"""
<<<<<<< HEAD
watershed_leaf_segmenter.py
============================
Definitive adaptive Watershed segmenter for diseased plant leaf images.

Built from iterative analysis and testing on 20 leaf disease types across
tomato, potato, and bell pepper plants, covering every major failure mode
encountered during development.

VALIDATED DISEASE / CONDITION TYPES
-------------------------------------
  Tomato   : healthy, early blight, late blight, bacterial spot, septoria
             leaf spot, leaf mold, mosaic virus, spot extreme
  Potato   : early blight (green), late blight (RS_LB)
  Bell pepper: bacterial spot (two severity levels)
  Severe   : tuff (wilted/curled), tuff2 (shrivelled brown, minimal contrast),
             tuff3 (dark necrotic + cast shadow), tuff4 (heavily brown blighted),
             tuff5 (deep-lobed, dark, strong shadows)

FAILURE MODES ADDRESSED
-------------------------

  1. DARK / NECROTIC TISSUE ERASED
     Root cause: saturation-based or hue-based gates miss near-achromatic
     dark-brown tissue (s_ch < 14, v_ch < 40).
     Fix A: Mahalanobis distance in LAB space — no hue ranges, captures
            deviation from THIS image's background regardless of leaf colour.
     Fix B: dark_tissue_boost — explicit +8 score boost for pixels with
            v_ch <= 40 (near-black necrosis, fungal mass) so they can never
            fall below the Otsu threshold.
     Fix C: shadow suppression V>80 guard — the suppression pass will never
            remove dark pixels even when they look background-like in LAB.

  2. SHADOW INCLUDED IN MASK (over-segmentation)
     Root cause: cast shadows on the background share low V + low S with
     dark necrotic tissue; MD-only approaches can't separate them.
     Fix A: texture map (local std-dev) — leaf tissue has micro-texture
            (veins, lesions, surface roughness); shadows are smooth. Added
            to the combined score so shadow pixels score lower.
     Fix B: conditional shadow suppression pass — fires only when coverage
            > shadow_trigger (default 0.65). Removes pixels that are both
            statistically close to background AND bright (V>80), so only
            genuine shadow pixels are removed, not dark leaf tissue.
     Fix C: robust border std (MAD) for shadow cutoff — prevents the cutoff
            being inflated by leaf-edge pixels that contaminated the border.

  3. UNDER-SEGMENTATION WHEN LEAF ≈ BACKGROUND COLOUR (e.g. tuff2)
     Root cause: a fixed score threshold fails when a dying brown leaf is
     nearly the same LAB colour as the gray-brown background.
     Fix: ADAPTIVE Otsu threshold on the score histogram. Otsu finds the
          valley between background and foreground peaks automatically,
          regardless of the absolute score values. Self-calibrates per image.

  4. BORDER CONTAMINATION IN BACKGROUND MODEL (e.g. potato shadow)
     Root cause: the border sampling ring sometimes captures leaf-edge
     pixels or the shadow cast by the leaf, inflating the background
     covariance and making the MD too permissive.
     Fix: MAD-based outlier rejection — border pixels more than 2.5 robust
          standard deviations from the border median are excluded before
          fitting the Gaussian background model. The remaining inliers are
          clean background samples.

  5. CHROMATIC BACKGROUNDS (tomato_leaf_mold blue bg, tomato_mosiac purple)
     Root cause: hue-based gates and saturation thresholds tuned for gray
     backgrounds fail on strongly tinted backgrounds.
     Fix: Mahalanobis distance in LAB inherently handles any background
          colour — the model is fitted to the specific background in the
          image, not a generic "background colour".

  6. HOLES IN MASK (disease lesion pits, deep crevices, Watershed boundary)
     Root cause: dark disease spots and folded crevices can score as
     background; Watershed boundary lines (-1) also create edge gaps.
     Fix: _fill_holes() applied twice — once before Watershed (on the rough
          mask) and once after (on the Watershed result). Interior components
          of the inverted mask that do not touch the image border are filled.

ALGORITHM PIPELINE
-------------------
  Step 1  Colour-space conversions (LAB, HSV, gray)
  Step 2  Robust background model
            - Sample border pixels (outer 6% ring)
            - Reject outlier border pixels using MAD (leaf-edge contamination)
            - Fit multivariate Gaussian to inlier border pixels in LAB space
            - Compute per-pixel Mahalanobis distance from background
  Step 3  Texture map  (local grayscale std-dev, 11x11 window)
  Step 4  Dark-tissue boost  (v_ch <= 40 -> +8 to score)
  Step 5  Combined score  =  MD + texture_weight * texture + dark_boost
  Step 6  Adaptive Otsu threshold on score histogram
  Step 7  Rough binary mask  +  CLOSE (bridge gaps) -> OPEN (remove specks)
  Step 8  Largest connected component selection
  Step 9  Conditional shadow suppression
            (only if cov > shadow_trigger AND pixel is bright AND near-BG in MD)
  Step 10 Interior hole fill (pre-Watershed)
  Step 11 Watershed markers from distance transform
  Step 12 cv2.watershed() boundary refinement
  Step 13 Interior hole fill (post-Watershed)

DEPENDENCIES
------------
    pip install opencv-python numpy

USAGE
-----
    # Single image
    python watershed_leaf_segmenter.py leaf.jpg

    # Single image to specific output folder
    python watershed_leaf_segmenter.py leaf.jpg ./output

    # Batch (all images in a folder)
    python watershed_leaf_segmenter.py ./images ./output

    # Background colour options
    python watershed_leaf_segmenter.py ./images ./output --bg white
    python watershed_leaf_segmenter.py ./images ./output --bg gray
    python watershed_leaf_segmenter.py ./images ./output --bg transparent  # BGRA PNG

    # Save binary mask alongside each result
    python watershed_leaf_segmenter.py ./images ./output --save_mask

    # Skip 4-panel comparison image
    python watershed_leaf_segmenter.py ./images ./output --no_comparison

TUNING GUIDE
------------
  Shadow or background still included:
    -> increase --texture_weight (try 6-8)
    -> decrease --shadow_trigger (try 0.55)

  Dark / brown tissue being cut away:
    -> decrease --texture_weight (try 2-3)
    -> increase --shadow_trigger (try 0.75)
    -> decrease --dark_boost_threshold (try 50)

  Very shrivelled / curled leaf (tuff2-type):
    -> Default settings handle this via adaptive Otsu. If still under-segmenting,
       decrease --texture_weight to 2.0 so the score relies more on pure MD.
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


# ============================================================================
# STEP 2: ROBUST BACKGROUND MODEL
# ============================================================================

def _robust_background_mahalanobis(lab, border_ratio=0.06):
    """
    Per-pixel Mahalanobis distance from a ROBUST border-sampled BG model.

    WHY MAHALANOBIS IN LAB?
    -----------------------
    LAB is perceptually uniform — equal distances correspond to equal
    perceived colour differences.  The Mahalanobis distance captures
    deviation across lightness (L), green/red (a), and yellow/blue (b)
    axes simultaneously.  This makes it robust to any background tint:
    warm concrete, cool gray felt, blue paper, or purple cloth.

    WHY ROBUST (MAD-BASED) FITTING?
    --------------------------------
    Simple mean+cov of border pixels is contaminated by leaf-edge pixels
    and cast shadow pixels that land within the border ring.  These inflate
    the covariance, making the MD permissive (everything looks far from BG).
    Rejecting border pixels more than 2.5 robust standard deviations from
    the border median gives clean background samples.

    Returns
    -------
    md      : float32 (H x W), higher = more different from background
    bg_mean : 3-vector, fitted background LAB mean (used by shadow pass)
    """
    h, w = lab.shape[:2]
    t = max(4, int(min(h, w) * border_ratio))

    border = np.zeros((h, w), dtype=bool)
    border[:t, :] = True;  border[-t:, :] = True
    border[:, :t] = True;  border[:, -t:] = True

    bg_px = lab[border].astype(np.float64)

    # MAD-based outlier rejection
    median  = np.median(bg_px, axis=0)
    mad     = np.median(np.abs(bg_px - median), axis=0) * 1.4826  # scale to sigma
    mad     = np.maximum(mad, 1.0)
    outlier_dist = np.max(np.abs(bg_px - median) / mad, axis=1)
    inliers = bg_px[outlier_dist < 2.5]
    if len(inliers) < 30:
        inliers = bg_px  # fallback if too many rejected

    bg_mean = inliers.mean(axis=0)
    bg_cov  = np.cov(inliers, rowvar=False) + np.eye(3) * 1e-6

    try:
        cov_inv = np.linalg.inv(bg_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(3)

    flat = lab.reshape(-1, 3).astype(np.float64)
    diff = flat - bg_mean
    md   = np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))
    return md.reshape(h, w).astype(np.float32), bg_mean


# ============================================================================
# STEP 3: TEXTURE MAP
# ============================================================================

def _texture_map(gray, kernel_size=11):
    """
    Local standard deviation (approximated via Gaussian blur variance).

    WHY TEXTURE?
    ------------
    Cast shadows share low V and low S with dark necrotic leaf tissue —
    colour-space features alone cannot separate them.  Leaf tissue has
    rich micro-texture (veins, lesions, surface roughness); a cast shadow
    is a smooth gradient.  Adding texture to the score means shadow pixels
    score lower than leaf-tissue pixels of equal colour.

    Returns
    -------
    texture : float32 (H x W), typical range 0-60
    """
    g    = gray.astype(np.float32)
    k    = (kernel_size, kernel_size)
    e_x2 = cv2.GaussianBlur(g * g, k, 0)
    ex_2 = cv2.GaussianBlur(g,     k, 0) ** 2
    return np.sqrt(np.maximum(e_x2 - ex_2, 0.0))


# ============================================================================
# HELPERS
# ============================================================================

def _fill_holes(mask):
    """
    Fill enclosed interior holes in a binary mask.

    Inverts the mask, labels connected components.  Any component that
    does NOT touch the image border is an interior hole (disease lesion
    pit, crevice, Watershed boundary line) and is filled.  Components
    touching the border are real background — left untouched.
    """
    h, w = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = mask.copy()
    for lid in range(1, n):
        x, y, ww, hh, _ = stats[lid]
        if x > 0 and y > 0 and (x + ww) < w and (y + hh) < h:
            filled[labels == lid] = 255
    return filled


def _largest_component(mask):
    """Keep only the single largest connected foreground component."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    lg = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == lg, 255, 0).astype(np.uint8)


# ============================================================================
# CORE SEGMENTATION
# ============================================================================

def segment_leaf(
    image,
    border_ratio=0.06,
    texture_weight=4.0,
    shadow_trigger=0.65,
    dark_boost_threshold=40,
    dark_boost_value=8.0,
):
    """
    Segment a single leaf image using the adaptive Watershed pipeline.

    Parameters
    ----------
    image                 : BGR uint8 numpy array
    border_ratio          : fraction of min(H,W) used as background sample ring
    texture_weight        : texture cue weight in combined score.
                            Raise (6-8) to suppress shadows.
                            Lower (2-3) to preserve very dark tissue.
    shadow_trigger        : coverage fraction above which shadow suppression
                            fires.  Lower (0.55) = more aggressive suppression.
    dark_boost_threshold  : HSV V value below which pixels receive dark_boost.
                            Raise to 50 if very dark tissue is being cut away.
    dark_boost_value      : score bonus for near-black pixels. Ensures
                            near-black necrosis always exceeds Otsu threshold.

    Returns
    -------
    mask      : uint8 binary mask (255=leaf, 0=background)
    segmented : BGR image with background set to black
    """
    h, w = image.shape[:2]

    # ── Step 1: Colour-space conversions ─────────────────────────────────────
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    # ── Step 2: Robust background model ──────────────────────────────────────
    md, bg_mean = _robust_background_mahalanobis(lab, border_ratio)

    # ── Step 3: Texture map ───────────────────────────────────────────────────
    texture = _texture_map(gray)

    # ── Step 4: Dark-tissue boost ─────────────────────────────────────────────
    # Near-black pixels (v_ch <= dark_boost_threshold) are almost certainly
    # leaf — either deep necrosis or fungal mass.  A plain background at this
    # darkness is extremely rare.  Boosting their score guarantees they survive
    # the Otsu threshold regardless of the image's contrast level.
    dark_boost = np.where(v_ch <= dark_boost_threshold,
                          dark_boost_value, 0.0).astype(np.float32)

    # ── Step 5: Combined score ────────────────────────────────────────────────
    score = md + (texture / 255.0) * texture_weight + dark_boost

    # ── Step 6: Adaptive Otsu threshold ──────────────────────────────────────
    # Otsu on the score histogram finds the valley between background and
    # foreground peaks automatically — self-calibrates to each image's
    # actual contrast level with no hardcoded values.
    score_8u     = (np.clip(score, 0, 30) / 30.0 * 255).astype(np.uint8)
    score_smooth = cv2.GaussianBlur(score_8u, (5, 5), 0)
    otsu_val, _  = cv2.threshold(score_smooth, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thr     = max(otsu_val / 255.0 * 30.0, 1.5)  # minimum guard

    # ── Step 7: Rough binary mask ─────────────────────────────────────────────
    rough = (score > otsu_thr).astype(np.uint8) * 255

    k7  = np.ones((7,  7),  np.uint8)
    k11 = np.ones((11, 11), np.uint8)

    # CLOSE first: bridges necrotic gaps and leaf folds
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k11, iterations=3)
    # OPEN after:  removes isolated background specks and debris
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN,  k7,  iterations=2)

    # ── Step 8: Largest connected component ───────────────────────────────────
    rough = _largest_component(rough)
    cov   = rough.sum() / 255 / (h * w)

    # ── Step 9: Conditional shadow suppression ────────────────────────────────
    # Triggered only when coverage exceeds shadow_trigger.
    # Removes pixels that satisfy ALL of:
    #   (a) statistically close to background (MD < robust_shadow_cut)
    #   (b) bright  (V > 80) — protects dark necrotic tissue unconditionally
    #   (c) currently in the rough mask
    if cov > shadow_trigger:
        border = np.zeros((h, w), dtype=bool)
        t = max(4, int(min(h, w) * border_ratio))
        border[:t, :] = True;  border[-t:, :] = True
        border[:, :t] = True;  border[:, -t:] = True

        # Robust border std via MAD (prevents inflated cutoff from leaf-edge px)
        border_md  = md[border]
        border_mad = np.median(np.abs(border_md - np.median(border_md))) * 1.4826
        shadow_cut = max(border_mad * 2.5, 1.5)

        shadow_px = (md < shadow_cut) & (v_ch > 80) & (rough > 0)
        rough[shadow_px] = 0

        rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k11, iterations=2)
        rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN,  k7,  iterations=1)
        rough = _largest_component(rough)

    # ── Step 10: Interior hole fill (pre-Watershed) ───────────────────────────
    filled = _fill_holes(rough)

    # ── Step 11: Watershed markers from distance transform ────────────────────
    # Distance transform of the filled mask acts as a height map.
    # Pixels far from any edge (high distance) -> sure foreground markers.
    # Threshold at 20% of max distance keeps thin / irregularly shaped leaves.
    dist   = cv2.distanceTransform(filled, cv2.DIST_L2, maskSize=5)
    dist_n = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg_f = cv2.threshold(dist_n, 0.20, 1.0, cv2.THRESH_BINARY)
    sure_fg      = np.uint8(sure_fg_f * 255)
    sure_bg      = cv2.dilate(filled, k7, iterations=3)
    unknown      = cv2.subtract(sure_bg, sure_fg)

    # Marker convention: 0=unknown, 1=background, 2+=foreground regions
    _, markers   = cv2.connectedComponents(sure_fg)
    markers      = markers + 1
    markers[unknown == 255] = 0

    # ── Step 12: Watershed boundary refinement ────────────────────────────────
    markers   = cv2.watershed(image.copy(), markers)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_mask[(markers > 1) & (markers != -1)] = 255

    # ── Step 13: Interior hole fill (post-Watershed) ──────────────────────────
    # Watershed boundary lines (-1) and dark disease lesions can leave
    # small enclosed holes at the leaf edge — fill them.
    leaf_mask = _fill_holes(leaf_mask)

    segmented = image.copy()
    segmented[leaf_mask == 0] = 0

    return leaf_mask, segmented


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def apply_mask(image, mask, bg_color="black"):
    """
    Composite the segmented leaf onto a chosen background.

    bg_color : "black" | "white" | "gray" | "transparent"
    "transparent" returns a 4-channel BGRA image (alpha = mask), suitable
    for use as input to further ML pipelines or compositing tools.
    """
    if bg_color == "transparent":
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        return bgra

    fills = {
        "black": np.zeros_like(image),
        "white": np.full_like(image, 255),
        "gray":  np.full_like(image, 180),
    }
    result             = image.copy()
    result[mask == 0]  = fills.get(bg_color, fills["black"])[mask == 0]
    return result


def build_comparison(image, mask, segmented, filename=""):
    """
    Build a 4-panel side-by-side strip:
    Original | Green mask overlay | Black bg | White bg
    """
    T = 256

    def prep(img):
        img = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (20, 20, 20), 1, cv2.LINE_AA)
        return out

    mask_vis              = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    mask_vis[mask > 0]    = (0, 210, 0)

    white_bg              = image.copy()
    white_bg[mask == 0]   = (255, 255, 255)

    seg_bgr = (cv2.cvtColor(segmented, cv2.COLOR_BGRA2BGR)
               if segmented.ndim == 4 else segmented)

    row = np.hstack([
        label(prep(image),    "Original"),
        label(prep(mask_vis), "Leaf mask"),
        label(prep(seg_bgr),  "Black bg"),
        label(prep(white_bg), "White bg"),
    ])

    ratio  = np.count_nonzero(mask) / mask.size * 100
    header = np.full((26, row.shape[1], 3), (215, 225, 210), dtype=np.uint8)
    cv2.putText(header,
                f"Watershed  |  {filename}  |  Leaf area: {ratio:.1f}%",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30),
                1, cv2.LINE_AA)
    return np.vstack([header, row])


# ============================================================================
# BATCH RUNNER
# ============================================================================

SUPPORTED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",
}


def process_single(image_path, output_dir, bg_color="black",
                   save_mask=False, save_comparison=True,
                   texture_weight=4.0, shadow_trigger=0.65,
                   dark_boost_threshold=40):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  x Could not load: {image_path.name}")
        return

    try:
        mask, segmented = segment_leaf(
            image,
            texture_weight=texture_weight,
            shadow_trigger=shadow_trigger,
            dark_boost_threshold=dark_boost_threshold,
        )
    except Exception as exc:
        print(f"  x Failed ({image_path.name}): {exc}")
        return

    ratio   = np.count_nonzero(mask) / mask.size * 100
    stem    = image_path.stem
    ext     = image_path.suffix.lower()
    out_ext = ".png" if bg_color == "transparent" else ext

    result = apply_mask(image, mask, bg_color)
    cv2.imwrite(str(output_dir / f"{stem}_segmented{out_ext}"), result,
                [cv2.IMWRITE_JPEG_QUALITY, 95] if "jpg" in out_ext else [])

    if save_mask:
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask)

    if save_comparison:
        seg_bgr = (cv2.cvtColor(segmented, cv2.COLOR_BGRA2BGR)
                   if bg_color == "transparent" else segmented)
        panel = build_comparison(image, mask, seg_bgr,
                                 filename=image_path.name)
        cv2.imwrite(str(output_dir / f"{stem}_comparison.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])

    print(f"  + {image_path.name:50s}  leaf area: {ratio:.1f}%")


def process_batch(input_dir, output_dir, bg_color="black",
                  save_mask=False, save_comparison=True,
                  texture_weight=4.0, shadow_trigger=0.65,
                  dark_boost_threshold=40):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [f for f in sorted(input_dir.iterdir())
              if f.suffix in SUPPORTED_EXTS]

    if not images:
        print(f"[WARNING] No supported images found in '{input_dir}'")
        return

    print(f"Processing {len(images)} image(s)  ->  '{output_dir}'\n")
    for p in images:
        process_single(
            p, output_dir,
            bg_color=bg_color, save_mask=save_mask,
            save_comparison=save_comparison,
            texture_weight=texture_weight,
            shadow_trigger=shadow_trigger,
            dark_boost_threshold=dark_boost_threshold,
        )
    print("\nDone.")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Adaptive Watershed leaf segmenter -- definitive version. "
            "Validated on 20 disease types across tomato, potato, bell pepper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tuning guide
------------
  Shadows / background still in mask:
    -> increase --texture_weight (try 6-8)
    -> decrease --shadow_trigger (try 0.55)

  Dark / brown / necrotic tissue cut away:
    -> decrease --texture_weight (try 2-3)
    -> increase --shadow_trigger (try 0.75)
    -> increase --dark_boost_threshold (try 50-60)

  Severely wilted / shrivelled leaf (tuff2-type):
    -> try --texture_weight 2.0 (relies more on pure MD for low-contrast leaves)

Examples
--------
  python watershed_leaf_segmenter.py leaf.jpg
  python watershed_leaf_segmenter.py ./images ./output --bg white --save_mask
  python watershed_leaf_segmenter.py ./images ./output --bg transparent
  python watershed_leaf_segmenter.py ./images ./output --texture_weight 6 --shadow_trigger 0.55
  python watershed_leaf_segmenter.py ./images ./output --dark_boost_threshold 55
        """,
    )
    p.add_argument("input",  help="Input image file or folder")
    p.add_argument("output", nargs="?", default=None,
                   help="Output folder (default: <input>/watershed_output)")
    p.add_argument("--bg",
                   choices=["black", "white", "gray", "transparent"],
                   default="black",
                   help="Background colour for segmented output (default: black)")
    p.add_argument("--save_mask",     action="store_true",
                   help="Also save a binary mask PNG for each image")
    p.add_argument("--no_comparison", action="store_true",
                   help="Skip the 4-panel comparison image")
    p.add_argument("--texture_weight", type=float, default=4.0,
                   help="Texture cue weight (default 4.0). Raise to suppress shadows.")
    p.add_argument("--shadow_trigger", type=float, default=0.65,
                   help="Coverage fraction that triggers shadow suppression (default 0.65).")
    p.add_argument("--dark_boost_threshold", type=int, default=40,
                   help="HSV-V threshold below which pixels get a score boost (default 40).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else \
          (inp if inp.is_dir() else inp.parent) / "watershed_output"
    out.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        bg_color             = args.bg,
        save_mask            = args.save_mask,
        save_comparison      = not args.no_comparison,
        texture_weight       = args.texture_weight,
        shadow_trigger       = args.shadow_trigger,
        dark_boost_threshold = args.dark_boost_threshold,
    )

    if inp.is_dir():
        process_batch(inp, out, **kwargs)
    elif inp.is_file() and inp.suffix in SUPPORTED_EXTS:
        process_single(inp, out, **kwargs)
    else:
        print(f"Error: '{inp}' is not a valid image file or folder.",
              file=sys.stderr)
        sys.exit(1)
=======
Leaf segmentation module — vegetation-index and color-space methods
(no GrabCut dependency).

Three independent approaches (no GrabCut dependency):

1. **Color-Index Segmentation (ExG / ExGR / CIVE)**
   Vegetation indices from precision agriculture literature.
   ExG  = 2g − r − b   (Woebbecke et al., 1995)
   ExGR = ExG − ExR     (Meyer & Neto, 2008)
   CIVE = 0.441r − 0.811g + 0.385b + 18.787 (Kataoka et al., 2003)

2. **LAB a*-channel Segmentation**
   The CIELAB a* channel encodes the green↔red opponent axis.
   Green leaves have strongly negative a* values, making thresholding
   straightforward and robust across illumination changes.

3. **SLIC Superpixel Segmentation**
   Over-segments the image into compact colour–spatial regions using
   Simple Linear Iterative Clustering, then classifies each superpixel
   as "leaf" or "background" based on its mean colour.

All three share the same SegmentationResult dataclass and morphological
post-processing from the original module, so they are drop-in
replacements for GrabCut.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class SegmentationMethod(Enum):
    """Leaf segmentation methods (non-GrabCut)."""
    COLOR_INDEX = "color_index"
    LAB_ASTAR = "lab_astar"
    SLIC_SUPERPIXEL = "slic_superpixel"
    FAILED = "failed"


@dataclass
class SegmentationResult:
    """Result of leaf segmentation (same interface as original module)."""
    success: bool
    method: str
    mask: Optional[np.ndarray]
    segmented: Optional[np.ndarray]
    mask_ratio: float
    error_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "method": self.method,
            "mask_ratio": self.mask_ratio,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# 1. Colour-Index Segmentation
# ---------------------------------------------------------------------------

class ColorIndexSegmenter:
    """
    Segment leaf regions using vegetation colour indices.

    Works by converting RGB to normalised rgb, computing ExG (or ExGR/CIVE),
    then thresholding with Otsu on the index image.
    """

    SUPPORTED_INDICES = ("exg", "exgr", "cive")

    def __init__(
        self,
        index: str = "exg",
        morph_kernel_size: int = 7,
        morph_open_iter: int = 2,
        morph_close_iter: int = 3,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
    ):
        if index.lower() not in self.SUPPORTED_INDICES:
            raise ValueError(f"Unsupported index '{index}'. Choose from {self.SUPPORTED_INDICES}")
        self.index = index.lower()
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.morph_open_iter = morph_open_iter
        self.morph_close_iter = morph_close_iter
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    # -- index computation ---------------------------------------------------

    @staticmethod
    def _normalise_rgb(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (r, g, b) in [0, 1] where r+g+b = 1 per pixel."""
        img = image.astype(np.float64)
        # OpenCV loads BGR
        B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        total = R + G + B + 1e-6  # avoid division by zero
        return R / total, G / total, B / total

    def compute_exg(self, image: np.ndarray) -> np.ndarray:
        """Excess Green Index: 2g − r − b."""
        r, g, b = self._normalise_rgb(image)
        return 2.0 * g - r - b

    def compute_exgr(self, image: np.ndarray) -> np.ndarray:
        """Excess Green minus Excess Red: (2g−r−b)−(1.4r−g)."""
        r, g, b = self._normalise_rgb(image)
        exg = 2.0 * g - r - b
        exr = 1.4 * r - g
        return exg - exr

    def compute_cive(self, image: np.ndarray) -> np.ndarray:
        """Colour Index of Vegetation Extraction."""
        r, g, b = self._normalise_rgb(image)
        # Lower CIVE → more vegetation  (original formula has positive r term)
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.787 / 255.0
        return -cive  # negate so higher = vegetation (consistent with ExG)

    def compute_index(self, image: np.ndarray) -> np.ndarray:
        """Compute the selected vegetation index."""
        if self.index == "exg":
            return self.compute_exg(image)
        elif self.index == "exgr":
            return self.compute_exgr(image)
        else:
            return self.compute_cive(image)

    # -- segmentation --------------------------------------------------------

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Run colour-index segmentation pipeline."""
        try:
            idx_img = self.compute_index(image)

            # Normalise to 0-255 for Otsu thresholding
            idx_norm = cv2.normalize(idx_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Otsu on the index image
            _, binary = cv2.threshold(idx_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Ensure leaf = white by checking border
            border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
            if np.mean(border) > 127:
                binary = cv2.bitwise_not(binary)

            # Morphological cleanup
            binary = self._morphology(binary)

            # Quality check
            ratio = np.sum(binary > 0) / binary.size
            if ratio < self.min_mask_ratio or ratio > self.max_mask_ratio:
                return SegmentationResult(
                    success=False,
                    method=SegmentationMethod.COLOR_INDEX.value,
                    mask=None, segmented=None, mask_ratio=ratio,
                    error_message=f"Mask ratio {ratio:.1%} out of range",
                )

            segmented = self._apply_mask(image, binary)
            return SegmentationResult(
                success=True,
                method=SegmentationMethod.COLOR_INDEX.value,
                mask=binary, segmented=segmented, mask_ratio=ratio,
            )
        except Exception as exc:
            logger.error(f"ColorIndex segmentation failed: {exc}")
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.COLOR_INDEX.value,
                mask=None, segmented=None, mask_ratio=0.0,
                error_message=str(exc),
            )

    # -- helpers --------------------------------------------------------------

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_open_iter)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_close_iter)
        return _keep_largest_component(closed)

    @staticmethod
    def _apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(image)
        result[mask > 0] = image[mask > 0]
        return result


# ---------------------------------------------------------------------------
# 2. LAB a*-channel Segmentation
# ---------------------------------------------------------------------------

class LABSegmenter:
    """
    Segment leaves using CIELAB colour channels.

    Uses a three-pronged approach to capture **all** leaf tissue — green,
    brown, dried, and necrotic — while rejecting neutral backgrounds:

    1. **Inverted a* channel + Otsu** — catches green tissue (low a*).
    2. **Chroma channel (√(a*² + b*²)) + Otsu** — catches colourful
       brown/red tissue that has high chroma.
    3. **b* channel + Otsu** — catches dried/tan tissue that has warm
       (positive) b* but may have low chroma because it is desaturated.

    The three binary masks are combined with OR.  After morphological
    cleanup, **interior holes are filled** so that any dried tissue
    enclosed within the leaf boundary is never excluded.
    """

    def __init__(
        self,
        morph_kernel_size: int = 7,
        morph_open_iter: int = 2,
        morph_close_iter: int = 3,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
        blur_ksize: int = 5,
    ):
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.morph_open_iter = morph_open_iter
        self.morph_close_iter = morph_close_iter
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.blur_ksize = blur_ksize

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Run LAB triple-channel segmentation with interior hole filling."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_channel = lab[:, :, 1].astype(np.float64)  # uint8: 128 = neutral
            b_channel = lab[:, :, 2].astype(np.float64)

            # --- Mask 1: inverted a* (captures green tissue) ---------------
            a_inv = (255 - lab[:, :, 1])  # uint8
            a_blurred = cv2.GaussianBlur(a_inv, (self.blur_ksize, self.blur_ksize), 0)
            _, bin_a = cv2.threshold(a_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # --- Mask 2: chroma (captures colourful brown/red tissue) ------
            # Chroma = sqrt((a*-128)² + (b*-128)²)  — distance from neutral
            chroma = np.sqrt((a_channel - 128) ** 2 + (b_channel - 128) ** 2)
            chroma_u8 = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            chroma_blurred = cv2.GaussianBlur(chroma_u8, (self.blur_ksize, self.blur_ksize), 0)
            _, bin_c = cv2.threshold(chroma_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # --- Mask 3: b* channel (captures dried/tan/warm tissue) -------
            # Dried brown leaves have positive b* (warm tone) even when
            # desaturated, while white/grey backgrounds sit near b*=128.
            b_u8 = lab[:, :, 2]  # already uint8
            b_blurred = cv2.GaussianBlur(b_u8, (self.blur_ksize, self.blur_ksize), 0)
            _, bin_b = cv2.threshold(b_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Border heuristic for each mask
            for bm in (bin_a, bin_c, bin_b):
                border = np.concatenate([bm[0], bm[-1], bm[:, 0], bm[:, -1]])
                if np.mean(border) > 127:
                    bm[:] = cv2.bitwise_not(bm)

            # Combine: union of all three masks
            binary = cv2.bitwise_or(bin_a, bin_c)
            binary = cv2.bitwise_or(binary, bin_b)

            # Morphological cleanup
            binary = self._morphology(binary)

            # --- Fill interior holes so dried inner tissue is included -----
            binary = self._fill_holes(binary)

            ratio = np.sum(binary > 0) / binary.size
            if ratio < self.min_mask_ratio or ratio > self.max_mask_ratio:
                return SegmentationResult(
                    success=False,
                    method=SegmentationMethod.LAB_ASTAR.value,
                    mask=None, segmented=None, mask_ratio=ratio,
                    error_message=f"Mask ratio {ratio:.1%} out of range",
                )

            segmented = np.zeros_like(image)
            segmented[binary > 0] = image[binary > 0]
            return SegmentationResult(
                success=True,
                method=SegmentationMethod.LAB_ASTAR.value,
                mask=binary, segmented=segmented, mask_ratio=ratio,
            )
        except Exception as exc:
            logger.error(f"LAB segmentation failed: {exc}")
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.LAB_ASTAR.value,
                mask=None, segmented=None, mask_ratio=0.0,
                error_message=str(exc),
            )

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_open_iter)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_close_iter)
        return _keep_largest_component(closed)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fill interior holes in the leaf mask.

        After finding the external contour of the leaf, all enclosed
        regions (holes caused by dried/brown tissue being missed) are
        filled in.  This ensures the full leaf area is captured.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return mask
        filled = mask.copy()
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled


# ---------------------------------------------------------------------------
# 3. SLIC Superpixel Segmentation
# ---------------------------------------------------------------------------

class SLICSegmenter:
    """
    Segment leaves via SLIC superpixels + colour classification.

    Steps:
    1. Compute SLIC superpixels (scikit-image)
    2. For each superpixel, compute mean LAB colour
    3. Classify as "green/leaf" if a* < threshold (green region)
    4. Build binary mask from leaf-classified superpixels
    5. Morphological cleanup + largest component
    """

    def __init__(
        self,
        n_segments: int = 200,
        compactness: float = 15.0,
        sigma: float = 1.0,
        a_star_threshold: int = 125,
        morph_kernel_size: int = 7,
        morph_open_iter: int = 1,
        morph_close_iter: int = 3,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
    ):
        """
        Args:
            n_segments: Approximate number of superpixels.
            compactness: SLIC compactness (higher = more square).
            sigma: Gaussian pre-smoothing sigma.
            a_star_threshold: LAB a* value below which a superpixel is
                              classified as "leaf" (128 = neutral; lower = greener).
            morph_*: morphological cleanup parameters.
            min/max_mask_ratio: QC bounds.
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.a_star_threshold = a_star_threshold
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.morph_open_iter = morph_open_iter
        self.morph_close_iter = morph_close_iter
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Run SLIC superpixel segmentation."""
        try:
            from skimage.segmentation import slic
            from skimage.color import rgb2lab
        except ImportError:
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.SLIC_SUPERPIXEL.value,
                mask=None, segmented=None, mask_ratio=0.0,
                error_message="scikit-image not installed (required for SLIC)",
            )

        try:
            # Convert BGR → RGB → float [0,1] for skimage
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

            # Compute superpixels
            labels = slic(
                rgb,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                start_label=0,
                channel_axis=-1,
            )

            # Convert to LAB for colour classification
            lab = rgb2lab(rgb)  # L [0,100], a [-128,128], b [-128,128]

            # Classify each superpixel
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for lbl in np.unique(labels):
                region = labels == lbl
                mean_a = np.mean(lab[region, 1])  # a* channel, signed
                # Negative a* → green → leaf
                # Convert threshold from uint8 convention (128=0) to signed:
                threshold_signed = self.a_star_threshold - 128  # e.g. 125 → -3
                if mean_a < threshold_signed:
                    mask[region] = 255

            # Morphological cleanup
            mask = self._morphology(mask)

            ratio = np.sum(mask > 0) / mask.size
            if ratio < self.min_mask_ratio or ratio > self.max_mask_ratio:
                return SegmentationResult(
                    success=False,
                    method=SegmentationMethod.SLIC_SUPERPIXEL.value,
                    mask=None, segmented=None, mask_ratio=ratio,
                    error_message=f"Mask ratio {ratio:.1%} out of range",
                )

            segmented = np.zeros_like(image)
            segmented[mask > 0] = image[mask > 0]
            return SegmentationResult(
                success=True,
                method=SegmentationMethod.SLIC_SUPERPIXEL.value,
                mask=mask, segmented=segmented, mask_ratio=ratio,
            )
        except Exception as exc:
            logger.error(f"SLIC segmentation failed: {exc}")
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.SLIC_SUPERPIXEL.value,
                mask=None, segmented=None, mask_ratio=0.0,
                error_message=str(exc),
            )

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_open_iter)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_close_iter)
        return _keep_largest_component(closed)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == largest] = 255
    return result
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
