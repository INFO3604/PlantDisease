"""Complete preprocessing pipeline -- DeepLabV3+ leaf segmentation + K-means disease detection.

Pipeline order:
    Resize (Lanczos)
    -> White-Balance (Gray-World)
    -> Denoise (Bilateral)
    -> Contrast (AGCWD)
    -> DeepLabV3+ Leaf Segmentation (feature-saliency + GrabCut + shadow removal)
    -> K-means L*a*b* Disease Detection (cluster on a*, b* + R-G auto-select)
    -> Colour-coded severity overlay with contour outlines

Disease detection adapted from:
    - GreenR (ashish-code/GreenR-visual-plant-necrosis-analysis): K-means on a*,b*
    - berk12cyr/Plant-Leaf-Disease-Detection-Segmentation: HSV hue gating
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .leaf_segmentation import (
    SegmentationMethod,
    SegmentationResult,
    WatershedSegmenter,
    DeepLabV3Segmenter,
    segment_leaf,
    apply_mask,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass for pipeline output
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full output of the preprocessing pipeline."""
    original: np.ndarray
    resized: np.ndarray
    white_balanced: np.ndarray
    denoised: np.ndarray
    contrast_enhanced: np.ndarray
    leaf_mask: Optional[np.ndarray]
    segmented_leaf: Optional[np.ndarray]
    disease_mask: Optional[np.ndarray]
    disease_overlay: Optional[np.ndarray]
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
    """End-to-end preprocessing pipeline.

    Default: DeepLabV3+ leaf segmentation + multi-scale disease detection.
    Watershed segmenter available as fallback.
    """

    SUPPORTED_SEG = ("watershed", "deeplabv3")

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        segmentation_method: str = "deeplabv3",
        disease_threshold: float = 2.5,
    ):
        if segmentation_method not in self.SUPPORTED_SEG:
            raise ValueError(
                f"Unknown segmentation method '{segmentation_method}'. "
                f"Choose from {self.SUPPORTED_SEG}"
            )
        self.target_size = target_size
        self.seg_method = segmentation_method
        self.disease_threshold = disease_threshold

        # Build leaf segmenter
        if segmentation_method == "watershed":
            self._segmenter = WatershedSegmenter()
        else:
            self._segmenter = DeepLabV3Segmenter()

    # ------------------------------------------------------------------
    # Individual preprocessing steps
    # ------------------------------------------------------------------

    def resize_lanczos(self, image: np.ndarray) -> np.ndarray:
        """Resize using Lanczos interpolation (preserves detail)."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def white_balance_gray_world(image: np.ndarray) -> np.ndarray:
        """Gray-World white-balance."""
        img = image.astype(np.float64)
        means = img.mean(axis=(0, 1))
        global_mean = means.mean()
        scale = global_mean / (means + 1e-6)
        return np.clip(img * scale, 0, 255).astype(np.uint8)

    @staticmethod
    def denoise_bilateral(image: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filter denoising."""
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    @staticmethod
    def enhance_contrast_agcwd(image: np.ndarray) -> np.ndarray:
        """Adaptive Gamma Correction with Weighting Distribution (AGCWD)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        v = hsv[:, :, 2]

        hist, _ = np.histogram(v, bins=256, range=(0, 256))
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)

        pdf_max = pdf.max()
        pdf_w = pdf_max * (1.0 - cdf)

        cdf_w = np.cumsum(pdf_w)
        cdf_w_norm = cdf_w / (cdf_w[-1] + 1e-6)

        lut = np.zeros(256, dtype=np.float64)
        for i in range(256):
            lut[i] = 255.0 * ((i / 255.0) ** (1.0 - cdf_w_norm[i]))

        v_new = lut[np.clip(v, 0, 255).astype(np.int32)]
        hsv[:, :, 2] = np.clip(v_new, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ------------------------------------------------------------------
    # Disease detection -- K-means L*a*b* clustering
    # ------------------------------------------------------------------

    def detect_disease_multiscale(
        self,
        image: np.ndarray,
        leaf_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        """Disease detection using K-means clustering in L*a*b* colour space.

        Adapted from GreenR (ashish-code/GreenR-visual-plant-necrosis-analysis)
        and berk12cyr/Plant-Leaf-Disease-Detection-Segmentation.

        Steps:
          1. Extract leaf pixels, convert to L*a*b*
          2. Cluster on a*, b* channels using KMeans (K=3)
          3. Identify "greenest" cluster (lowest mean R-G in RGB space)
          4. Mark remaining clusters as disease if mean R-G > 0
          5. Confirm via HSV warm-hue gate (hue 0-30 or 155-180, or dark necrotic)
          6. Shadow rejection + morphological cleanup
        """
        valid = leaf_mask > 0
        total_leaf = int(np.sum(valid))
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        if total_leaf < 100:
            return empty, 0.0, 0, total_leaf

        h, w = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- K-means on a*, b* within leaf mask ---
        ab_pixels = lab[valid, 1:3].reshape(-1, 2).astype(np.float32)
        n_clusters = min(3, len(ab_pixels))
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        labels = km.fit_predict(ab_pixels)

        label_map = np.full((h, w), -1, dtype=np.int32)
        label_map[valid] = labels

        # --- Score each cluster by mean(R) - mean(G) ---
        R_ch = image[:, :, 2].astype(np.float64)
        G_ch = image[:, :, 1].astype(np.float64)

        rg_scores = []
        for k in range(n_clusters):
            mask_k = label_map == k
            n_px = int(np.sum(mask_k))
            if n_px < 10:
                rg_scores.append((-999.0, n_px))
                continue
            rg = float(np.mean(R_ch[mask_k]) - np.mean(G_ch[mask_k]))
            rg_scores.append((rg, n_px))

        # Identify greenest cluster (lowest R-G = most green)
        greenest_k = min(
            range(n_clusters), key=lambda k: rg_scores[k][0]
        )
        greenest_rg = rg_scores[greenest_k][0]

        # Disease = non-greenest clusters where R-G is positive OR
        # significantly higher than the greenest cluster (catches dark spots)
        disease_map = np.zeros((h, w), dtype=np.uint8)
        for k in range(n_clusters):
            if k == greenest_k:
                continue
            rg_score, n_px = rg_scores[k]
            if rg_score > 0 or rg_score > greenest_rg + 15:
                disease_map[label_map == k] = 255

        if np.sum(disease_map > 0) == 0:
            return empty, 0.0, 0, total_leaf

        # --- HSV confirmation gate ---
        h_ch = hsv[:, :, 0].astype(np.float64)
        s_ch = hsv[:, :, 1].astype(np.float64)
        v_ch = hsv[:, :, 2].astype(np.float64)

        warm_hue = (h_ch <= 35) | (h_ch >= 155)       # Red-orange-yellow-brown
        dark_necrotic = (v_ch < 100) & (s_ch < 80)     # Dark / dead tissue
        disease_map[~(warm_hue | dark_necrotic)] = 0

        # --- Shadow rejection ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kk = (11, 11)
        e_x2 = cv2.GaussianBlur(gray * gray, kk, 0)
        ex_2 = cv2.GaussianBlur(gray, kk, 0) ** 2
        texture = np.sqrt(np.maximum(e_x2 - ex_2, 0.0))

        shadow = (texture < 3.0) & (v_ch < 60) & (s_ch < 50) & (disease_map > 0)
        disease_map[shadow] = 0

        very_dark = (v_ch < 30) & (texture < 3.0) & (disease_map > 0)
        disease_map[very_dark] = 0

        very_bright = (v_ch > 220) & (s_ch < 30) & (disease_map > 0)
        disease_map[very_bright] = 0

        # --- Morphological cleanup ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        disease_map = cv2.morphologyEx(disease_map, cv2.MORPH_OPEN, kernel, iterations=1)
        disease_map = cv2.morphologyEx(disease_map, cv2.MORPH_CLOSE, kernel, iterations=1)
        disease_map = cv2.bitwise_and(disease_map, leaf_mask)

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
        """Colour-coded disease overlay with contour outlines.

        Categories (6-level severity palette):
          - Dark necrotic    -> Deep Red
          - Brown / blight   -> Bright Orange
          - Light brown / tan -> Warm Amber
          - Yellow chlorotic  -> Bright Yellow
          - Red / purple spot -> Magenta
          - Other diseased    -> Dark Crimson
        """
        overlay = image.copy()
        mask_bool = disease_mask > 0
        if not np.any(mask_bool):
            return overlay

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        a_ch = lab[:, :, 1]

        # Yellow / chlorotic
        yellow = mask_bool & (h >= 15) & (h <= 35) & (s > 40)

        # Light brown / tan
        light_brown = mask_bool & (h >= 8) & (h < 22) & (s > 25) & (v > 80)
        light_brown = light_brown & ~yellow

        # Dark brown / necrotic
        dark_brown = mask_bool & (
            ((h < 22) & (v <= 80)) | ((h < 15) & (s <= 50))
        )
        dark_brown = dark_brown & ~yellow & ~light_brown

        # Red / purple spots (high a* in LAB)
        red_purple = mask_bool & (a_ch > 145) & (s > 50)
        red_purple = red_purple & ~yellow & ~light_brown & ~dark_brown

        # Other
        other = mask_bool & ~yellow & ~light_brown & ~dark_brown & ~red_purple

        # Apply colour tints (BGR)
        tints = [
            (dark_brown,  np.array([30, 30, 255])),     # Deep Red
            (light_brown, np.array([0, 140, 255])),      # Bright Orange
            (yellow,      np.array([0, 255, 255])),      # Bright Yellow
            (red_purple,  np.array([180, 50, 200])),     # Magenta
            (other,       np.array([30, 0, 180])),       # Dark Crimson
        ]
        for cond, colour in tints:
            if np.any(cond):
                overlay[cond] = (
                    (1 - alpha) * overlay[cond].astype(np.float64)
                    + alpha * colour.astype(np.float64)
                ).astype(np.uint8)

        # White contour outlines
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
        """Execute the full preprocessing pipeline."""
        steps = []

        # 1. Resize
        resized = self.resize_lanczos(image)
        steps.append("resize_lanczos")

        # 2. White-balance
        wb = self.white_balance_gray_world(resized)
        steps.append("white_balance_gray_world")

        # 3. Denoise
        denoised = self.denoise_bilateral(wb)
        steps.append("denoise_bilateral")

        # 4. Contrast
        enhanced = self.enhance_contrast_agcwd(denoised)
        steps.append("contrast_agcwd")

        # 5. Leaf segmentation
        seg_result: SegmentationResult = self._segmenter.segment(enhanced)
        steps.append(f"segment_{self.seg_method}")

        if not seg_result.success:
            logger.warning(f"Segmentation failed: {seg_result.error_message}")
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
                segmentation_method=seg_result.method,
                segmentation_success=False,
                mask_ratio=seg_result.mask_ratio,
                steps_applied=steps,
            )

        leaf_mask = seg_result.mask
        segmented = seg_result.segmented

        # 6. Disease detection (multi-scale U-Net-inspired)
        disease_mask, severity, diseased_px, total_px = (
            self.detect_disease_multiscale(segmented, leaf_mask)
        )
        steps.append("disease_kmeans_lab")

        # 7. Overlay
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
            segmentation_method=seg_result.method,
            segmentation_success=True,
            mask_ratio=seg_result.mask_ratio,
            severity_percent=severity,
            diseased_pixels=diseased_px,
            total_leaf_pixels=total_px,
            steps_applied=steps,
        )
