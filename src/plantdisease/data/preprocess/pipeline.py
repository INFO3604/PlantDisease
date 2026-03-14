"""Complete preprocessing pipeline with SAM-first leaf isolation.

Pipeline order:
    Resize (Lanczos)
    -> White-Balance (Gray-World)
    -> Denoise (Bilateral)
    -> Contrast (AGCWD)
    -> SAM full-leaf isolation (fallback to classical watershed leaf segmenter)
    -> Apply leaf mask to AGCWD image
    -> Convert masked leaf to HSV/LAB
    -> Watershed disease segmentation inside leaf mask only
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .leaf_segmentation import segment_leaf

logger = logging.getLogger(__name__)

# --- Optional SAM initialisation ---------------------------------------------
try:
    from .sam_segmenation import load_sam, segment_leaf_sam

    try:
        sam_generator = load_sam()
    except Exception as _e:
        logger.warning(f"Failed to initialise SAM generator: {_e}")
        sam_generator = None
except Exception:
    sam_generator = None


def _mask_quality(mask: Optional[np.ndarray], min_ratio: float = 0.05, max_ratio: float = 0.95) -> bool:
    """Quick quality check for a binary mask: existence + reasonable coverage."""
    if mask is None:
        return False
    if not isinstance(mask, np.ndarray):
        return False
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return False
    ratio = float(np.sum(mask > 0)) / (h * w)
    return (ratio >= min_ratio) and (ratio <= max_ratio)


# ---------------------------------------------------------------------------
# Dataclass for pipeline output
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full output of the preprocessing pipeline."""
    # Images
    original: np.ndarray
    resized: np.ndarray
    white_balanced: np.ndarray
    denoised: np.ndarray
    contrast_enhanced: np.ndarray
    leaf_mask: Optional[np.ndarray]
    segmented_leaf: Optional[np.ndarray]
    disease_mask: Optional[np.ndarray]
    disease_overlay: Optional[np.ndarray]
    # Metadata
    segmentation_method: str = ""
    segmentation_success: bool = False
    mask_ratio: float = 0.0
    severity_percent: float = 0.0
    diseased_pixels: int = 0
    total_leaf_pixels: int = 0
    steps_applied: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "segmentation_method": self.segmentation_method,
            "segmentation_success": self.segmentation_success,
            "mask_ratio": round(self.mask_ratio, 4),
            "severity_percent": round(self.severity_percent, 2),
            "diseased_pixels": self.diseased_pixels,
            "total_leaf_pixels": self.total_leaf_pixels,
            "steps_applied": self.steps_applied,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    target_size : tuple
        (width, height) for resizing.
    segmentation_method : str
        Preserved for backward compatibility. Classical segmenter is used only
        as fallback when SAM is unavailable or low-quality.
    disease_threshold : float
        Retained for compatibility; not used by watershed disease segmentation.
    """

    SUPPORTED_SEG = ("color_index", "lab_astar", "slic_superpixel")

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        segmentation_method: str = "lab_astar",
        color_index: str = "exg",
        disease_threshold: float = 2.5,
        sam_generator: Optional[object] = None,
        sam_enabled: bool = True,
    ):
        if segmentation_method not in self.SUPPORTED_SEG:
            raise ValueError(
                f"Unknown segmentation method '{segmentation_method}'. "
                f"Choose from {self.SUPPORTED_SEG}"
            )
        self.target_size = target_size
        self.seg_method = segmentation_method
        self.color_index = color_index
        self.disease_threshold = disease_threshold

        # SAM fallback control (lazy-loadable)
        self.sam_enabled = sam_enabled
        # Allow user to pass a preloaded generator; otherwise defer initialisation
        self.sam_generator = sam_generator if sam_generator is not None else globals().get("sam_generator", None)

    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply binary mask to image, zeroing out background."""
        segmented = np.zeros_like(image)
        segmented[mask > 0] = image[mask > 0]
        return segmented

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def resize_lanczos(self, image: np.ndarray) -> np.ndarray:
        """Resize using Lanczos interpolation (preserves detail)."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def white_balance_gray_world(image: np.ndarray) -> np.ndarray:
        """Gray-World white-balance: scale each channel so its mean = global mean."""
        img = image.astype(np.float64)
        means = img.mean(axis=(0, 1))
        global_mean = means.mean()
        scale = global_mean / (means + 1e-6)
        balanced = np.clip(img * scale, 0, 255).astype(np.uint8)
        return balanced

    @staticmethod
    def denoise_bilateral(image: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filter denoising.

        Bilateral filtering smooths flat regions while keeping edges
        sharp — ideal for leaf images where disease boundaries matter.
        """
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    @staticmethod
    def enhance_contrast_agcwd(image: np.ndarray) -> np.ndarray:
        """
        Adaptive Gamma Correction with Weighting Distribution (AGCWD).

        Adjusts the gamma curve based on the PDF/CDF of the luminance histogram
        so that dark images get brightened more than bright images.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        v = hsv[:, :, 2]

        # Probability density & cumulative distribution
        hist, _ = np.histogram(v, bins=256, range=(0, 256))
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)

        # Weighting: pdf_w = pdf_max * (1 − cdf)
        pdf_max = pdf.max()
        pdf_w = pdf_max * (1.0 - cdf)

        # Normalised CDF of weighted PDF
        cdf_w = np.cumsum(pdf_w)
        cdf_w_norm = cdf_w / (cdf_w[-1] + 1e-6)

        # Adaptive gamma mapping
        lut = np.zeros(256, dtype=np.float64)
        for i in range(256):
            lut[i] = 255.0 * ((i / 255.0) ** (1.0 - cdf_w_norm[i]))

        v_new = lut[np.clip(v, 0, 255).astype(np.int32)]
        hsv[:, :, 2] = np.clip(v_new, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _ensure_sam(self) -> bool:
        """Ensure `self.sam_generator` is available when `sam_enabled`.

        Returns True when a usable generator is available.
        """
        if not self.sam_enabled:
            return False
        if self.sam_generator is not None:
            return True
        try:
            from .sam_segmenation import load_sam  # local import to avoid heavy load at module import
        except Exception as e:
            logger.debug(f"SAM module not importable: {e}")
            return False
        try:
            self.sam_generator = load_sam()
            return True
        except Exception as e:
            logger.warning(f"Failed to load SAM generator lazily: {e}")
            self.sam_enabled = False
            return False

    def detect_disease_watershed(
        self,
        image: np.ndarray,
        leaf_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        """Segment disease regions via marker-controlled watershed.

        Watershed runs strictly inside the provided leaf mask. Seed markers
        are built from Hue (HSV) and a* (LAB) channels.
        """
        valid = leaf_mask > 0
        total_leaf = int(np.sum(valid))
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        if total_leaf < 50:
            return empty, 0.0, 0, total_leaf

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        a_ch = lab[:, :, 1]

        a_vals = a_ch[valid].astype(np.float32)
        if a_vals.size < 50:
            return empty, 0.0, 0, total_leaf

        a_hi = float(np.percentile(a_vals, 68))
        a_lo = float(np.percentile(a_vals, 42))

        warm_hue = ((h_ch >= 4) & (h_ch <= 38) & (s_ch >= 28))
        a_seed = a_ch >= a_hi
        seed_fg = (warm_hue | a_seed) & valid

        seed_bg = (a_ch <= a_lo) & (~warm_hue) & valid

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seed_fg_u8 = (seed_fg.astype(np.uint8) * 255)
        seed_bg_u8 = (seed_bg.astype(np.uint8) * 255)
        seed_fg_u8 = cv2.morphologyEx(seed_fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        seed_fg_u8 = cv2.erode(seed_fg_u8, kernel, iterations=1)
        seed_bg_u8 = cv2.morphologyEx(seed_bg_u8, cv2.MORPH_OPEN, kernel, iterations=1)

        n_fg, fg_labels = cv2.connectedComponents(seed_fg_u8)

        markers = np.zeros(image.shape[:2], dtype=np.int32)
        markers[~valid] = 1
        markers[seed_bg_u8 > 0] = 1
        if n_fg > 1:
            markers[fg_labels > 0] = fg_labels[fg_labels > 0] + 1

        # If foreground seeds are weak, fall back to thresholding within leaf.
        if n_fg <= 1:
            disease_mask = seed_fg_u8
        else:
            ws_img = image.copy()
            ws_markers = cv2.watershed(ws_img, markers)
            disease_mask = np.zeros_like(seed_fg_u8)
            disease_mask[(ws_markers > 1) & valid] = 255

        disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        diseased = int(np.sum(disease_mask > 0))
        severity = (diseased / total_leaf * 100) if total_leaf > 0 else 0.0
        return disease_mask, severity, diseased, total_leaf

    # ------------------------------------------------------------------
    # Disease detection (Mahalanobis)
    # ------------------------------------------------------------------

    def detect_disease_mahalanobis(
        self,
        image: np.ndarray,
        leaf_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Mahalanobis-distance disease detection.

        Healthy (green) pixels define the reference distribution.  Pixels
        whose Mahalanobis distance exceeds `self.disease_threshold` are
        classified as diseased.

        Returns
        -------
        disease_mask : np.ndarray
        severity_percent : float
        diseased_pixels : int
        total_leaf_pixels : int
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)

        # Collect healthy (green) reference pixels
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
        green_leaf = cv2.bitwise_and(green_mask, leaf_mask)

        green_pixels = lab[green_leaf > 0]
        if len(green_pixels) < 50:
            # Not enough green — use all leaf pixels as reference
            green_pixels = lab[leaf_mask > 0]

        if len(green_pixels) < 10:
            # Cannot compute distribution
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            return empty, 0.0, 0, int(np.sum(leaf_mask > 0))

        mean = np.mean(green_pixels, axis=0)
        cov = np.cov(green_pixels, rowvar=False)
        try:
            cov_inv = np.linalg.inv(cov + np.eye(3) * 1e-6)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(3)

        # Compute Mahalanobis distance for all leaf pixels
        leaf_coords = np.argwhere(leaf_mask > 0)
        leaf_lab = lab[leaf_mask > 0]
        diff = leaf_lab - mean
        mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        # Threshold
        disease_map = np.zeros(image.shape[:2], dtype=np.uint8)
        diseased_idx = mahal > self.disease_threshold
        disease_map[leaf_coords[diseased_idx, 0], leaf_coords[diseased_idx, 1]] = 255

        # ----------------------------------------------------------
        # Shadow rejection: remove pixels that are shadows, not disease.
        # Shadows are characterised by LOW value (dark) AND low-to-
        # moderate saturation.  Real disease (brown spots, necrosis)
        # tends to retain moderate-high saturation even when dark.
        # We also reject very dark pixels that lack colour content.
        # ----------------------------------------------------------
        shadow_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),       # any hue, low S, low V
            np.array([179, 70, 85]),    # S < 70 AND V < 85 → shadow
        )
        # Also reject near-black pixels on the leaf (V < 40)
        very_dark = hsv[:, :, 2] < 40
        shadow_mask[very_dark] = 255
        # Remove shadow pixels from disease map
        disease_map = cv2.bitwise_and(disease_map, cv2.bitwise_not(shadow_mask))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        disease_map = cv2.morphologyEx(disease_map, cv2.MORPH_OPEN, kernel)
        disease_map = cv2.morphologyEx(disease_map, cv2.MORPH_CLOSE, kernel)
        # Re-apply leaf mask to prevent overflow
        disease_map = cv2.bitwise_and(disease_map, leaf_mask)

        total_leaf = int(np.sum(leaf_mask > 0))
        diseased = int(np.sum(disease_map > 0))
        severity = (diseased / total_leaf * 100) if total_leaf > 0 else 0.0

        return disease_map, severity, diseased, total_leaf

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_disease_overlay(
        image: np.ndarray,
        disease_mask: np.ndarray,
        alpha: float = 0.75,
    ) -> np.ndarray:
        """Bold color-coded overlay on diseased regions with contour outlines.

        Classifies each diseased pixel by its original colour and tints it
        with vivid, easy-to-distinguish warm colours:
          - Dark brown / necrotic  → Deep Red
          - Light brown / tan      → Bright Orange
          - Yellow / chlorotic     → Bright Yellow
          - Other diseased         → Dark Crimson

        Also draws white contour outlines around each diseased region
        for maximum visibility.
        """
        overlay = image.copy()
        mask_bool = disease_mask > 0
        if not np.any(mask_bool):
            return overlay

        # Convert to HSV for colour classification
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]  # 0-179
        s = hsv[:, :, 1]  # 0-255
        v = hsv[:, :, 2]  # 0-255

        # --- Classify diseased pixels by their original colour ---
        # Yellow / chlorotic: warm hue 15-35 with some saturation
        yellow_cond = mask_bool & (h >= 15) & (h <= 35) & (s > 40)

        # Light brown / tan:  hue 8-22, moderate+ brightness
        light_brown_cond = mask_bool & (h >= 8) & (h < 22) & (s > 25) & (v > 80)
        light_brown_cond = light_brown_cond & ~yellow_cond

        # Dark brown / necrotic: dark or desaturated warm pixels
        dark_brown_cond = mask_bool & (
            ((h < 22) & (v <= 80)) |
            ((h < 15) & (s <= 50))
        )
        dark_brown_cond = dark_brown_cond & ~yellow_cond & ~light_brown_cond

        # Everything else
        other_cond = mask_bool & ~yellow_cond & ~light_brown_cond & ~dark_brown_cond

        # --- Apply BOLD colour fills (BGR) ---
        # Dark brown → Bright Red  (30, 30, 255)
        if np.any(dark_brown_cond):
            overlay[dark_brown_cond] = (
                (1 - alpha) * overlay[dark_brown_cond].astype(np.float64)
                + alpha * np.array([30, 30, 255], dtype=np.float64)
            ).astype(np.uint8)

        # Light brown → Bright Orange  (0, 140, 255)
        if np.any(light_brown_cond):
            overlay[light_brown_cond] = (
                (1 - alpha) * overlay[light_brown_cond].astype(np.float64)
                + alpha * np.array([0, 140, 255], dtype=np.float64)
            ).astype(np.uint8)

        # Yellow → Bright Yellow  (0, 255, 255)
        if np.any(yellow_cond):
            overlay[yellow_cond] = (
                (1 - alpha) * overlay[yellow_cond].astype(np.float64)
                + alpha * np.array([0, 255, 255], dtype=np.float64)
            ).astype(np.uint8)

        # Other → Dark Crimson  (30, 0, 180)
        if np.any(other_cond):
            overlay[other_cond] = (
                (1 - alpha) * overlay[other_cond].astype(np.float64)
                + alpha * np.array([30, 0, 180], dtype=np.float64)
            ).astype(np.uint8)

        # --- Draw white contour outlines for extra clarity ---
        contours, _ = cv2.findContours(
            disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay

    @staticmethod
    def create_leaf_overlay(
        image: np.ndarray,
        leaf_mask: np.ndarray,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Green-tinted overlay on leaf region."""
        overlay = image.copy()
        mask_bool = leaf_mask > 0
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool].astype(np.float64)
            + alpha * np.array([0, 200, 0], dtype=np.float64)
        ).astype(np.uint8)
        return overlay

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> PipelineResult:
        """
        Execute the full preprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format.

        Returns
        -------
        PipelineResult
        """
        steps = []

        # 1. Resize (Lanczos)
        resized = self.resize_lanczos(image)
        steps.append("resize_lanczos")

        # 2. White-balance (Gray-World)
        wb = self.white_balance_gray_world(resized)
        steps.append("white_balance_gray_world")

        # 3. Denoise (Bilateral Filter — edge-preserving)
        denoised = self.denoise_bilateral(wb)
        steps.append("denoise_bilateral")

        # 4. Contrast (AGCWD)
        enhanced = self.enhance_contrast_agcwd(denoised)
        steps.append("contrast_agcwd")

        # 5. SAM full-leaf isolation from AGCWD output (primary path)
        leaf_mask = None
        segmentation_method = ""

        if self._ensure_sam() and self.sam_generator is not None:
            try:
                leaf_mask = segment_leaf_sam(enhanced, self.sam_generator)
            except BaseException as e:
                logger.warning(f"SAM segmentation raised an error: {e}")
                leaf_mask = None

        if _mask_quality(leaf_mask):
            segmentation_method = "sam"
            steps.append("segment_sam_primary")
        else:
            # Fallback to classical leaf segmenter if SAM is unavailable/weak.
            try:
                classical_mask, _ = segment_leaf(enhanced)
                if _mask_quality(classical_mask):
                    leaf_mask = classical_mask
                    segmentation_method = "classical_watershed_fallback"
                    steps.append("segment_classical_fallback")
            except Exception as e:
                logger.warning(f"Classical segmentation fallback failed: {e}")

        # If still no valid leaf mask, fail gracefully
        if leaf_mask is None or not _mask_quality(leaf_mask):
            logger.warning("Segmentation failed: no usable leaf mask")
            return PipelineResult(
                original=image,
                resized=resized,
                white_balanced=wb,
                denoised=denoised,
                contrast_enhanced=enhanced,
                leaf_mask=None,
                segmented_leaf=None,
                disease_mask=None,
                disease_overlay=None,
                segmentation_method=segmentation_method,
                segmentation_success=False,
                mask_ratio=0.0,
                steps_applied=steps,
            )

        # 6. Apply SAM/classical mask to AGCWD image (background -> 0)
        segmented = self.apply_mask(enhanced, leaf_mask)
        steps.append("apply_leaf_mask_to_agcwd")

        # 7. Convert masked leaf to HSV/LAB (inputs for marker construction)
        _ = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
        _ = cv2.cvtColor(segmented, cv2.COLOR_BGR2LAB)
        steps.append("convert_masked_leaf_to_hsv_lab")

        # 8. Watershed disease segmentation inside leaf mask only
        disease_mask, severity, diseased_px, total_px = self.detect_disease_watershed(
            segmented, leaf_mask
        )
        steps.append("disease_watershed_inside_leaf_mask")

        # 7. Overlay disease regions on the segmented leaf
        overlay = self.create_disease_overlay(segmented, disease_mask)
        steps.append("overlay_on_segmented")

        return PipelineResult(
            original=image,
            resized=resized,
            white_balanced=wb,
            denoised=denoised,
            contrast_enhanced=enhanced,
            leaf_mask=leaf_mask,
            segmented_leaf=segmented,
            disease_mask=disease_mask,
            disease_overlay=overlay,
            segmentation_method=segmentation_method,
            segmentation_success=True,
            mask_ratio=float(np.sum(leaf_mask > 0)) / (leaf_mask.shape[0] * leaf_mask.shape[1]),
            severity_percent=severity,
            diseased_pixels=diseased_px,
            total_leaf_pixels=total_px,
            steps_applied=steps,
        )
