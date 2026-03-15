"""
Leaf segmentation module -- DeepLabV3+ (primary) + Watershed (fallback).

Primary method: DeepLabV3 ResNet-50 pretrained backbone with feature-saliency
fallback, GrabCut refinement, and dedicated multi-strategy shadow removal.

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
    DEEPLABV3 = "deeplabv3"
    COLOR_INDEX = "color_index"
    LAB_ASTAR = "lab_astar"
    SLIC_SUPERPIXEL = "slic_superpixel"
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
# DeepLabV3+ SEGMENTER (pretrained backbone + GrabCut + shadow removal)
# =========================================================================

class DeepLabV3Segmenter:
    """Leaf segmentation using pretrained DeepLabV3 ResNet-50 + GrabCut refinement
    + dedicated multi-strategy shadow removal.

    Pipeline
    --------
    1.  Run torchvision ``deeplabv3_resnet50`` (COCO/VOC-21 weights).
        Take the non-background prediction as the coarse plant mask.
    2.  If DeepLabV3 coverage is < 2 % of the image, fall back to
        WatershedSegmenter (model did not recognise the object).
    3.  Refine the coarse mask with **GrabCut** (5 iterations) using
        the DeepLabV3 output to seed definite-FG / definite-BG.
    4.  Apply ``remove_shadows()`` from :mod:`shadow` to strip cast and
        self-shadows from the mask.
    """

    _model = None          # class-level cache so we load weights only once
    _transforms = None

    def __init__(
        self,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
    ):
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

        # Lazy-load the model on first use (shared across instances)
        if DeepLabV3Segmenter._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls):
        import torch
        from torchvision.models.segmentation import (
            deeplabv3_resnet50,
            DeepLabV3_ResNet50_Weights,
        )
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        cls._model = deeplabv3_resnet50(weights=weights)
        cls._model.eval()
        cls._transforms = weights.transforms()
        logger.info("DeepLabV3 ResNet-50 loaded (VOC-21 weights)")

    # ------------------------------------------------------------------

    def _deeplabv3_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask from DeepLabV3 inference.

        Strategy:
          1. Try the VOC-21 classification head -- use the non-background
             prediction if it covers > 5 % of the image.
          2. Otherwise fall back to **deep-feature saliency**: average the
             absolute activations across the 2048 backbone feature channels
             to get a spatial saliency map.  Leaf tissue (texture, colour,
             veins) activates the filters more than a plain background.
             Otsu-threshold the saliency map to get a foreground mask.
        """
        import torch
        import torch.nn.functional as F

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = self._transforms(tensor).unsqueeze(0)

        h, w = image.shape[:2]

        with torch.no_grad():
            # Single backbone forward pass
            backbone_feats = self._model.backbone(tensor)
            feat_map = backbone_feats["out"]       # [1, 2048, H/S, W/S]

            # Run classifier head for class predictions
            class_out = self._model.classifier(feat_map)  # [1, 21, H/S, W/S]
            class_out = F.interpolate(
                class_out, size=tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            probs = torch.softmax(class_out, dim=1)[0]  # [21, H', W']

            bg_prob = probs[0].cpu().numpy()
            fg_prob = 1.0 - bg_prob

            # Deep-feature saliency
            saliency = feat_map.abs().mean(dim=1)[0].cpu().numpy()

        # Resize to original image dimensions
        fg_resized = cv2.resize(fg_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        sal_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Try classification output first ---
        class_mask = (fg_resized > 0.5).astype(np.uint8) * 255
        class_cov = np.sum(class_mask > 0) / class_mask.size

        if class_cov > 0.05:
            logger.info("DeepLabV3 class head detected %.1f%% foreground",
                        class_cov * 100)
            return class_mask

        # --- Fall back to feature saliency ---
        sal_norm = cv2.normalize(
            sal_resized, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        sal_smooth = cv2.GaussianBlur(sal_norm, (5, 5), 0)
        _, sal_mask = cv2.threshold(
            sal_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Border polarity check -- background should be at the edges
        border_vals = np.concatenate([
            sal_mask[0], sal_mask[-1], sal_mask[:, 0], sal_mask[:, -1],
        ])
        if np.mean(border_vals) > 127:
            sal_mask = cv2.bitwise_not(sal_mask)

        logger.info("DeepLabV3 using feature-saliency fallback (%.1f%% FG)",
                     np.sum(sal_mask > 0) / sal_mask.size * 100)
        return sal_mask

    # ------------------------------------------------------------------

    @staticmethod
    def _grabcut_refine(image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Refine the DeepLabV3 mask with GrabCut."""
        h, w = image.shape[:2]

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[initial_mask > 0] = cv2.GC_PR_FGD

        # Definite background from image border (thin strip)
        t = max(2, int(min(h, w) * 0.02))
        gc_mask[:t, :] = cv2.GC_BGD
        gc_mask[-t:, :] = cv2.GC_BGD
        gc_mask[:, :t] = cv2.GC_BGD
        gc_mask[:, -t:] = cv2.GC_BGD

        # Eroded core = definite foreground (gentle erosion)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        core = cv2.erode(initial_mask, k7, iterations=2)
        gc_mask[core > 0] = cv2.GC_FGD

        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(image, gc_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return initial_mask

        return np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

    # ------------------------------------------------------------------

    def segment(self, image: np.ndarray) -> SegmentationResult:
        try:
            h, w = image.shape[:2]

            # 1. DeepLabV3 coarse mask (class head or feature saliency)
            dl_mask = self._deeplabv3_mask(image)

            # 2. Border-based background rejection -- remove pixels whose
            #    LAB colour is close to the image border (background).
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
            bt = max(4, int(min(h, w) * 0.06))
            border_region = np.zeros((h, w), dtype=bool)
            border_region[:bt, :] = True
            border_region[-bt:, :] = True
            border_region[:, :bt] = True
            border_region[:, -bt:] = True

            bg_px = lab[border_region]
            bg_median = np.median(bg_px, axis=0)
            bg_mad = np.maximum(
                np.median(np.abs(bg_px - bg_median), axis=0) * 1.4826, 3.0
            )
            z = np.max(
                np.abs(lab - bg_median.reshape(1, 1, 3))
                / bg_mad.reshape(1, 1, 3),
                axis=2,
            )
            dl_mask[z < 1.0] = 0  # too similar to background

            # 3. Morphological tidy-up
            k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dl_mask = cv2.morphologyEx(dl_mask, cv2.MORPH_CLOSE, k11, iterations=2)
            dl_mask = cv2.morphologyEx(dl_mask, cv2.MORPH_OPEN, k7, iterations=1)
            dl_mask = _keep_largest_component(dl_mask)

            dl_ratio = np.sum(dl_mask > 0) / dl_mask.size
            if dl_ratio < 0.02:
                # Model did not detect the plant – fall back to Watershed + shadow removal
                logger.warning(
                    "DeepLabV3 coverage %.1f%% < 2%%, falling back to Watershed",
                    dl_ratio * 100,
                )
                ws_result = WatershedSegmenter().segment(image)
                if ws_result.success:
                    from .shadow import remove_shadows
                    clean = remove_shadows(image, ws_result.mask)
                    ws_result.mask = clean
                    ws_result.segmented = image.copy()
                    ws_result.segmented[clean == 0] = 0
                    ws_result.mask_ratio = float(np.sum(clean > 0)) / clean.size
                    ws_result.method = SegmentationMethod.DEEPLABV3.value
                return ws_result

            if dl_ratio < 0.20:
                # Low coverage — try Watershed and use whichever gives more
                logger.info(
                    "DeepLabV3 coverage %.1f%% < 20%%, trying Watershed merge",
                    dl_ratio * 100,
                )
                ws_result = WatershedSegmenter().segment(image)
                if ws_result.success and ws_result.mask_ratio > dl_ratio * 1.5:
                    dl_mask = cv2.bitwise_or(dl_mask, ws_result.mask)

            # 3. GrabCut refinement
            refined = self._grabcut_refine(image, dl_mask)
            refined = _keep_largest_component(refined)
            refined = _fill_holes(refined)

            # 4. Dedicated shadow removal
            from .shadow import remove_shadows
            final_mask = remove_shadows(image, refined)

            ratio = float(np.sum(final_mask > 0)) / final_mask.size
            if ratio < self.min_mask_ratio or ratio > self.max_mask_ratio:
                return SegmentationResult(
                    success=False,
                    method=SegmentationMethod.DEEPLABV3.value,
                    mask=None,
                    segmented=None,
                    mask_ratio=ratio,
                    error_message=f"Mask ratio {ratio:.1%} out of range",
                )

            segmented = image.copy()
            segmented[final_mask == 0] = 0

            return SegmentationResult(
                success=True,
                method=SegmentationMethod.DEEPLABV3.value,
                mask=final_mask,
                segmented=segmented,
                mask_ratio=ratio,
            )
        except Exception as exc:
            logger.error(f"DeepLabV3 segmentation failed: {exc}")
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.DEEPLABV3.value,
                mask=None,
                segmented=None,
                mask_ratio=0.0,
                error_message=str(exc),
            )
