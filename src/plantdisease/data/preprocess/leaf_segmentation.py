"""
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
