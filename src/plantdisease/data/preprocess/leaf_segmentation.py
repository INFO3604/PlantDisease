"""
Leaf segmentation module -- pure GrabCut with colour-based seed detection.

Primary method: GrabCut segmentation with automatic seed initialisation from
colour vegetation detection (ExG + HSV green + LAB background distance).
Uses 8 GrabCut iterations with 4-zone mask init for precise leaf outlines.

Fallback method: Adaptive Watershed segmenter with border-sampled background
model, texture analysis, and shadow suppression.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers & types
# ---------------------------------------------------------------------------

class SegmentationMethod(Enum):
    """Leaf segmentation methods."""
    WATERSHED = "watershed"
    GRABCUT = "grabcut"
    FAILED = "failed"


@dataclass
class SegmentationResult:
    """Result of leaf segmentation."""
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


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the single largest connected foreground component."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    lg = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == lg, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed interior holes in a binary mask."""
    h, w = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = mask.copy()
    for lid in range(1, n):
        x, y, ww, hh, _ = stats[lid]
        if x > 0 and y > 0 and (x + ww) < w and (y + hh) < h:
            filled[labels == lid] = 255
    return filled


# =========================================================================
# WATERSHED SEGMENTER (PRIMARY -- recommended)
# =========================================================================

def _robust_background_mahalanobis(lab: np.ndarray, border_ratio: float = 0.06):
    """Per-pixel Mahalanobis distance from a robust border-sampled BG model.

    Uses MAD-based outlier rejection on border pixels to prevent leaf-edge
    contamination from inflating the background covariance.
    """
    h, w = lab.shape[:2]
    t = max(4, int(min(h, w) * border_ratio))

    border = np.zeros((h, w), dtype=bool)
    border[:t, :] = True;  border[-t:, :] = True
    border[:, :t] = True;  border[:, -t:] = True

    bg_px = lab[border].astype(np.float64)

    # MAD-based outlier rejection
    median = np.median(bg_px, axis=0)
    mad = np.median(np.abs(bg_px - median), axis=0) * 1.4826
    mad = np.maximum(mad, 1.0)
    outlier_dist = np.max(np.abs(bg_px - median) / mad, axis=1)
    inliers = bg_px[outlier_dist < 2.5]
    if len(inliers) < 30:
        inliers = bg_px

    bg_mean = inliers.mean(axis=0)
    bg_cov = np.cov(inliers, rowvar=False) + np.eye(3) * 1e-6

    try:
        cov_inv = np.linalg.inv(bg_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(3)

    flat = lab.reshape(-1, 3).astype(np.float64)
    diff = flat - bg_mean
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
    return md.reshape(h, w).astype(np.float32), bg_mean


def _texture_map(gray: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Local standard deviation -- leaf tissue has texture, shadows are smooth."""
    g = gray.astype(np.float32)
    k = (kernel_size, kernel_size)
    e_x2 = cv2.GaussianBlur(g * g, k, 0)
    ex_2 = cv2.GaussianBlur(g, k, 0) ** 2
    return np.sqrt(np.maximum(e_x2 - ex_2, 0.0))


def segment_leaf(
    image: np.ndarray,
    border_ratio: float = 0.06,
    texture_weight: float = 4.0,
    shadow_trigger: float = 0.45,
    dark_boost_threshold: int = 40,
    dark_boost_value: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment leaf using adaptive Watershed pipeline.

    Returns (mask, segmented) where mask is uint8 binary (255=leaf).
    """
    h, w = image.shape[:2]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    md, bg_mean = _robust_background_mahalanobis(lab, border_ratio)
    texture = _texture_map(gray)

    dark_boost = np.where(
        v_ch <= dark_boost_threshold, dark_boost_value, 0.0
    ).astype(np.float32)

    score = md + (texture / 255.0) * texture_weight + dark_boost

    score_8u = (np.clip(score, 0, 30) / 30.0 * 255).astype(np.uint8)
    score_smooth = cv2.GaussianBlur(score_8u, (5, 5), 0)
    otsu_val, _ = cv2.threshold(
        score_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_thr = max(otsu_val / 255.0 * 30.0, 1.5)

    rough = (score > otsu_thr).astype(np.uint8) * 255

    k7 = np.ones((7, 7), np.uint8)
    k11 = np.ones((11, 11), np.uint8)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k11, iterations=3)
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, k7, iterations=2)
    rough = _keep_largest_component(rough)
    cov = rough.sum() / 255 / (h * w)

    if cov > shadow_trigger:
        border = np.zeros((h, w), dtype=bool)
        t = max(4, int(min(h, w) * border_ratio))
        border[:t, :] = True;  border[-t:, :] = True
        border[:, :t] = True;  border[:, -t:] = True

        border_md = md[border]
        border_mad = np.median(np.abs(border_md - np.median(border_md))) * 1.4826
        shadow_cut = max(border_mad * 2.5, 1.5)

        shadow_px = (md < shadow_cut) & (texture < 5.0) & (rough > 0)
        rough[shadow_px] = 0

        rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k11, iterations=2)
        rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, k7, iterations=1)
        rough = _keep_largest_component(rough)

    # --- ExG + chromaticity shadow rejection (always applied) ---
    # ExG is shadow-invariant: leaf tissue has positive ExG, shadows ~0
    img_f = image.astype(np.float64)
    B_ch, G_ch, R_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    total_rgb = R_ch + G_ch + B_ch + 1e-6
    exg = 2.0 * (G_ch / total_rgb) - (R_ch / total_rgb) - (B_ch / total_rgb)

    # Non-green smooth pixels are shadow / background leak
    shadow_exg = (exg < 0.05) & (texture < 6.0) & (rough > 0)
    rough[shadow_exg] = 0

    # Pixels whose a*,b* chromaticity matches background (shadow = same
    # chromaticity as background, just darker -- L is ignored)
    ab_pixel = lab[:, :, 1:].astype(np.float64)
    bg_ab = bg_mean[1:].reshape(1, 1, 2)
    ab_dist = np.sqrt(np.sum((ab_pixel - bg_ab) ** 2, axis=2))
    chroma_shadow = (ab_dist < 8.0) & (texture < 5.0) & (rough > 0)
    rough[chroma_shadow] = 0

    if np.any(shadow_exg | chroma_shadow):
        rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k11, iterations=2)
        rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, k7, iterations=1)
        rough = _keep_largest_component(rough)

    filled = _fill_holes(rough)

    dist = cv2.distanceTransform(filled, cv2.DIST_L2, maskSize=5)
    dist_n = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg_f = cv2.threshold(dist_n, 0.20, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg_f * 255)
    sure_bg = cv2.dilate(filled, k7, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image.copy(), markers)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_mask[(markers > 1) & (markers != -1)] = 255
    leaf_mask = _fill_holes(leaf_mask)

    segmented = image.copy()
    segmented[leaf_mask == 0] = 0

    return leaf_mask, segmented


def apply_mask(image: np.ndarray, mask: np.ndarray, bg_color: str = "black") -> np.ndarray:
    """Apply binary mask with chosen background."""
    if bg_color == "transparent":
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        return bgra
    fills = {
        "black": np.zeros_like(image),
        "white": np.full_like(image, 255),
        "gray": np.full_like(image, 180),
    }
    result = image.copy()
    result[mask == 0] = fills.get(bg_color, fills["black"])[mask == 0]
    return result


class WatershedSegmenter:
    """Wrapper providing a .segment() method for the pipeline."""

    def __init__(
        self,
        texture_weight: float = 4.0,
        shadow_trigger: float = 0.45,
        dark_boost_threshold: int = 40,
        dark_boost_value: float = 8.0,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
    ):
        self.texture_weight = texture_weight
        self.shadow_trigger = shadow_trigger
        self.dark_boost_threshold = dark_boost_threshold
        self.dark_boost_value = dark_boost_value
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def segment(self, image: np.ndarray) -> SegmentationResult:
        try:
            mask, segmented = segment_leaf(
                image,
                texture_weight=self.texture_weight,
                shadow_trigger=self.shadow_trigger,
                dark_boost_threshold=self.dark_boost_threshold,
                dark_boost_value=self.dark_boost_value,
            )
            ratio = float(np.sum(mask > 0)) / mask.size
            if ratio < self.min_mask_ratio or ratio > self.max_mask_ratio:
                return SegmentationResult(
                    success=False,
                    method=SegmentationMethod.WATERSHED.value,
                    mask=None, segmented=None, mask_ratio=ratio,
                    error_message=f"Mask ratio {ratio:.1%} out of range",
                )
            return SegmentationResult(
                success=True,
                method=SegmentationMethod.WATERSHED.value,
                mask=mask, segmented=segmented, mask_ratio=ratio,
            )
        except Exception as exc:
            logger.error(f"Watershed segmentation failed: {exc}")
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.WATERSHED.value,
                mask=None, segmented=None, mask_ratio=0.0,
                error_message=str(exc),
            )


# =========================================================================
# GRABCUT SEGMENTER (colour-seeded GrabCut for precise leaf outlines)
# =========================================================================

class GrabCutSegmenter:
    """Pure GrabCut leaf segmentation with colour-based seed detection.

    Pipeline
    --------
    1.  Build a coarse foreground seed from colour analysis:
        a) ExG (excess green) vegetation index — illumination-normalised
        b) HSV green detection (hue 25-85)
        c) HSV warm-hue detection (brown/yellow/red diseased tissue)
        d) LAB-based background distance (Mahalanobis from border pixels)
    2.  Combine colour seeds and clean up morphologically.
    3.  Run **GrabCut** (5 iterations, 10 px margin) with 4-zone init:
        - GC_BGD:    border strip (10 px, definite background)
        - GC_FGD:    deeply-eroded core of seed (definite foreground)
        - GC_PR_FGD: seed mask minus core
        - GC_PR_BGD: everything else
    4.  Post-process: keep largest component, fill holes, contour smoothing.
    5.  Apply ``remove_shadows()`` to strip cast shadows.
    """

    def __init__(
        self,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
    ):
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    # ------------------------------------------------------------------
    # Seed generation: colour-based foreground estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_seed_mask(image: np.ndarray) -> np.ndarray:
        """Build a coarse foreground seed mask from colour analysis.

        Combines four complementary signals:
          1. ExG > 0 (positive excess-green = vegetation)
          2. HSV green band (hue 25-85, sat > 30, val > 30)
          3. HSV warm band (hue 0-25, sat > 40, val > 40) for diseased tissue
          4. LAB background distance — pixels far from the border mean
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # --- Signal 1: ExG vegetation index ---
        img_f = image.astype(np.float64)
        B, G, R = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        total = R + G + B + 1e-6
        exg = 2.0 * (G / total) - (R / total) - (B / total)
        exg_mask = (exg > 0.0).astype(np.uint8) * 255

        # --- Signal 2: HSV green ---
        hsv_green = cv2.inRange(hsv, np.array([25, 30, 30]),
                                     np.array([85, 255, 255]))

        # --- Signal 3: HSV warm (diseased tissue) ---
        hsv_warm = cv2.inRange(hsv, np.array([0, 40, 40]),
                                    np.array([25, 255, 255]))

        # --- Signal 4: LAB background distance ---
        # Sample border pixels, compute Mahalanobis distance
        bt = max(4, int(min(h, w) * 0.06))
        border = np.zeros((h, w), dtype=bool)
        border[:bt, :] = True
        border[-bt:, :] = True
        border[:, :bt] = True
        border[:, -bt:] = True

        bg_px = lab[border].astype(np.float64)
        # MAD-based outlier rejection for robust BG model
        bg_median = np.median(bg_px, axis=0)
        bg_mad = np.median(np.abs(bg_px - bg_median), axis=0) * 1.4826
        bg_mad = np.maximum(bg_mad, 1.0)
        outlier = np.max(np.abs(bg_px - bg_median) / bg_mad, axis=1)
        inliers = bg_px[outlier < 2.5]
        if len(inliers) < 30:
            inliers = bg_px

        bg_mean = inliers.mean(axis=0)
        bg_cov = np.cov(inliers, rowvar=False) + np.eye(3) * 1e-6
        try:
            cov_inv = np.linalg.inv(bg_cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(3)

        flat = lab.reshape(-1, 3).astype(np.float64)
        diff = flat - bg_mean
        md = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
        md_map = md.reshape(h, w)

        # Otsu threshold on Mahalanobis distance
        md_8u = (np.clip(md_map, 0, 20) / 20.0 * 255).astype(np.uint8)
        md_smooth = cv2.GaussianBlur(md_8u, (5, 5), 0)
        _, md_mask = cv2.threshold(
            md_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # --- Combine all signals ---
        combined = cv2.bitwise_or(exg_mask, hsv_green)
        combined = cv2.bitwise_or(combined, hsv_warm)
        combined = cv2.bitwise_or(combined, md_mask)

        # Morphological cleanup
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k11, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k5, iterations=1)
        combined = _keep_largest_component(combined)
        combined = _fill_holes(combined)

        return combined

    # ------------------------------------------------------------------
    # GrabCut with 4-zone initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _grabcut_segment(image: np.ndarray, seed_mask: np.ndarray,
                         margin: int = 10, n_iter: int = 5) -> np.ndarray:
        """Run GrabCut with 4-zone initialisation.


        Zones:
          GC_BGD    -- definite background (border strip = margin px)
          GC_FGD    -- definite foreground (deeply eroded seed core)
          GC_PR_FGD -- probable foreground (seed mask minus core)
          GC_PR_BGD -- probable background (everything else)
        """
        h, w = image.shape[:2]

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Probable foreground from seed
        gc_mask[seed_mask > 0] = cv2.GC_PR_FGD

        # Definite background: border strip
        t = margin
        gc_mask[:t, :] = cv2.GC_BGD
        gc_mask[-t:, :] = cv2.GC_BGD
        gc_mask[:, :t] = cv2.GC_BGD
        gc_mask[:, -t:] = cv2.GC_BGD

        # Definite foreground: deeply-eroded core of seed mask
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        core = cv2.erode(seed_mask, k_erode, iterations=3)
        if np.sum(core > 0) > 0.02 * h * w:
            gc_mask[core > 0] = cv2.GC_FGD
        else:
            core = cv2.erode(seed_mask, k_erode, iterations=1)
            if np.sum(core > 0) > 0.01 * h * w:
                gc_mask[core > 0] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model,
                        n_iter, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            logger.warning("GrabCut failed, returning seed mask")
            return seed_mask

        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return result

    # ------------------------------------------------------------------
    # Contour smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_contour(mask: np.ndarray) -> np.ndarray:
        """Smooth the mask boundary using contour approximation + redraw."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return mask

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        smooth = np.zeros_like(mask)
        cv2.drawContours(smooth, [approx], -1, 255, cv2.FILLED)
        return smooth

    # ------------------------------------------------------------------
    # Main segment method
    # ------------------------------------------------------------------

    def segment(self, image: np.ndarray) -> SegmentationResult:
        try:
            h, w = image.shape[:2]

            # 1. Build colour-based seed mask
            seed = self._build_seed_mask(image)
            seed_cov = np.sum(seed > 0) / seed.size

            if seed_cov < 0.02:
                logger.warning(
                    "Seed coverage %.1f%% too low, falling back to Watershed",
                    seed_cov * 100,
                )
                ws_result = WatershedSegmenter().segment(image)
                if ws_result.success:
                    ws_result.method = SegmentationMethod.GRABCUT.value
                return ws_result

            # 2. GrabCut segmentation
            gc_mask = self._grabcut_segment(image, seed)
            gc_mask = _keep_largest_component(gc_mask)
            gc_mask = _fill_holes(gc_mask)

            # 3. Contour smoothing for cleaner outline
            gc_mask = self._smooth_contour(gc_mask)

            # 4. Final morphological smoothing
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, k5, iterations=1)

            # 5. Shadow removal
            from .shadow import remove_shadows
            final_mask = remove_shadows(image, gc_mask)

            ratio = float(np.sum(final_mask > 0)) / final_mask.size
            if ratio < self.min_mask_ratio or ratio > self.max_mask_ratio:
                return SegmentationResult(
                    success=False,
                    method=SegmentationMethod.GRABCUT.value,
                    mask=None,
                    segmented=None,
                    mask_ratio=ratio,
                    error_message=f"Mask ratio {ratio:.1%} out of range",
                )

            segmented = image.copy()
            segmented[final_mask == 0] = 0

            return SegmentationResult(
                success=True,
                method=SegmentationMethod.GRABCUT.value,
                mask=final_mask,
                segmented=segmented,
                mask_ratio=ratio,
            )
        except Exception as exc:
            logger.error(f"GrabCut segmentation failed: {exc}")
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.GRABCUT.value,
                mask=None,
                segmented=None,
                mask_ratio=0.0,
                error_message=str(exc),
            )


# Backward-compatible alias
DeepLabV3Segmenter = GrabCutSegmenter
