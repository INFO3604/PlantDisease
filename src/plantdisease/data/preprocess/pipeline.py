"""Preprocessing pipeline.

Pipeline order:
    1. Remove Background  (rembg deep-learning model → RGBA)
    2. Resize to 300×300   (Lanczos interpolation)
    3. Shadow Removal      (HSV colour-space thresholds)
    4. HSV Segmentation    (yellow / brown disease masks + adjacent green)
    5. Severity Calculation
    6. Data Normalisation  (pixel values scaled to [0, 1])
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .background import remove_background_rembg, get_rembg_session
from .shadow import remove_shadows_hsv_threshold
from .disease import DiseaseSegmenter, SeverityMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass for pipeline output
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full output of the preprocessing pipeline."""
    original: np.ndarray
    background_removed: np.ndarray          # RGBA with transparent bg
    resized: np.ndarray                     # BGR resized to target dims
    leaf_mask: Optional[np.ndarray]         # Binary mask from alpha channel
    shadow_removed: Optional[np.ndarray]    # After shadow removal
    shadow_mask: Optional[np.ndarray]       # Shadow pixels identified
    disease_mask: Optional[np.ndarray]      # Combined disease mask
    yellow_mask: Optional[np.ndarray]       # Yellow / chlorosis regions
    brown_mask: Optional[np.ndarray]        # Brown / necrosis regions
    disease_overlay: Optional[np.ndarray]   # Visualisation overlay
    normalized: Optional[np.ndarray] = None # float32 [0,1] for model input
    severity_percent: float = 0.0
    diseased_pixels: int = 0
    total_leaf_pixels: int = 0
    yellow_pixels: int = 0
    brown_pixels: int = 0
    steps_applied: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "severity_percent": round(self.severity_percent, 2),
            "diseased_pixels": self.diseased_pixels,
            "total_leaf_pixels": self.total_leaf_pixels,
            "yellow_pixels": self.yellow_pixels,
            "brown_pixels": self.brown_pixels,
            "steps_applied": self.steps_applied,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline.

    Steps
    -----
    1. Remove Background   – rembg (deep-learning, RGBA output)
    2. Resize              – Lanczos to *target_size*
    3. Shadow Removal      – HSV thresholds → bitwise AND w/ inverse mask
    4. HSV Segmentation    – yellow / brown thresholds + adjacent-green dilation
    5. Severity Ratio      – diseased pixels / total leaf pixels
    6. Normalisation       – pixel values to [0, 1]
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (300, 300),
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self._disease_segmenter = DiseaseSegmenter()
        self._rembg_session = get_rembg_session()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> PipelineResult:
        """Execute the full preprocessing pipeline on a single BGR image."""
        steps: list[str] = []

        # 1. Remove background (rembg) → RGBA with transparent bg
        bg_removed_rgba = remove_background_rembg(image, session=self._rembg_session)
        steps.append("remove_background_rembg")

        # 2. Resize (Lanczos) – operates on RGBA to keep alpha
        resized_rgba = cv2.resize(
            bg_removed_rgba, self.target_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        steps.append(f"resize_{self.target_size[0]}x{self.target_size[1]}")

        # Derive leaf mask from alpha channel
        alpha = resized_rgba[:, :, 3]
        leaf_mask = (alpha > 127).astype(np.uint8) * 255

        # Convert to BGR for colour processing
        bgr = cv2.cvtColor(resized_rgba, cv2.COLOR_RGBA2BGR)

        # 3. Shadow removal (HSV thresholds)
        shadow_removed, shadow_mask = remove_shadows_hsv_threshold(
            bgr, leaf_mask=leaf_mask,
        )
        steps.append("shadow_removal_hsv")

        # Refine leaf mask: exclude shadow pixels
        effective_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(shadow_mask))

        # 4. HSV disease segmentation
        disease_mask, yellow_mask, brown_mask, metrics = (
            self._disease_segmenter.segment_disease(
                shadow_removed, leaf_mask=effective_mask,
            )
        )
        steps.append("hsv_disease_segmentation")

        # 5. Severity (already computed inside DiseaseSegmenter)
        steps.append("severity_calculation")

        # Overlay for visualisation
        overlay = self._disease_segmenter.create_disease_overlay(
            shadow_removed, disease_mask, yellow_mask, brown_mask,
        )

        # 6. Normalisation
        normalized = None
        if self.normalize:
            normalized = shadow_removed.astype(np.float32) / 255.0
            steps.append("normalization")

        return PipelineResult(
            original=image,
            background_removed=resized_rgba,
            resized=bgr,
            leaf_mask=leaf_mask,
            shadow_removed=shadow_removed,
            shadow_mask=shadow_mask,
            disease_mask=disease_mask,
            yellow_mask=yellow_mask,
            brown_mask=brown_mask,
            disease_overlay=overlay,
            normalized=normalized,
            severity_percent=metrics.severity_percent,
            diseased_pixels=metrics.diseased_pixels,
            total_leaf_pixels=metrics.total_leaf_pixels,
            yellow_pixels=metrics.yellow_pixels,
            brown_pixels=metrics.brown_pixels,
            steps_applied=steps,
        )
