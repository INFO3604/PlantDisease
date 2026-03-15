"""
Shadow detection and removal for plant disease leaf images.

Provides ``remove_shadows()`` which strips obvious shadow pixels from a
binary leaf mask.  Uses conservative thresholds to avoid stripping real
leaf tissue -- it is better to keep a small shadow remnant than to
remove part of the leaf.
"""

import cv2
import numpy as np


def _texture_map(gray: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Local standard deviation -- leaf texture vs smooth shadow."""
    g = gray.astype(np.float32)
    k = (kernel_size, kernel_size)
    e_x2 = cv2.GaussianBlur(g * g, k, 0)
    ex_2 = cv2.GaussianBlur(g, k, 0) ** 2
    return np.sqrt(np.maximum(e_x2 - ex_2, 0.0))


def remove_shadows(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove obvious shadow pixels from a binary leaf mask.

    Uses two conservative strategies that require strong evidence before
    removing any pixel.  A global guard prevents the removal of more than
    30 % of the original mask (indicating the method is confused).

    Strategies
    ----------
    1. **Dark + smooth + non-green** -- pixels that are locally much darker
       than the surrounding leaf, have low texture (no veins / lesions), and
       are not green (low ExG).  All conditions must be met simultaneously.
    2. **Deep shadow** -- very dark AND desaturated AND smooth pixels.

    A safety guard protects clearly-leaf pixels (green, bright, textured,
    or saturated) from ever being removed.

    Parameters
    ----------
    image : np.ndarray  (BGR, uint8)
    mask  : np.ndarray  (uint8 binary, 255 = foreground)

    Returns
    -------
    np.ndarray  -- refined uint8 binary mask (255 = leaf only).
    """
    h, w = mask.shape[:2]
    original_count = int(np.sum(mask > 0))
    if original_count < 100:
        return mask

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = _texture_map(gray)

    v_ch = hsv[:, :, 2].astype(np.float64)
    s_ch = hsv[:, :, 1].astype(np.float64)
    l_ch = lab[:, :, 0].astype(np.float64)

    # ExG vegetation index
    img_f = image.astype(np.float64)
    B, G, R = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    total = R + G + B + 1e-6
    exg = 2.0 * (G / total) - (R / total) - (B / total)

    # Local L-channel deficit (shadow is locally darker)
    mask_f = (mask > 0).astype(np.float32)
    k_size = max(31, min(h, w) // 6)
    if k_size % 2 == 0:
        k_size += 1
    local_l_sum = cv2.GaussianBlur(
        l_ch.astype(np.float32) * mask_f, (k_size, k_size), 0
    )
    local_count = cv2.GaussianBlur(mask_f, (k_size, k_size), 0) + 1e-6
    local_mean_l = local_l_sum / local_count
    l_deficit = local_mean_l - l_ch.astype(np.float32)

    # --- Shadow detection (conservative -- all conditions AND'd) ---
    shadow = np.zeros((h, w), dtype=bool)

    # Type 1: Locally dark + smooth + non-green + dim
    shadow |= (
        (l_deficit > 25)
        & (texture < 4.0)
        & (exg < 0.02)
        & (v_ch < 80)
        & (mask > 0)
    )

    # Type 2: Very dark + desaturated (deep shadow)
    shadow |= (
        (v_ch < 40)
        & (s_ch < 50)
        & (texture < 3.0)
        & (mask > 0)
    )

    # --- Safety: NEVER remove clearly-leaf pixels ---
    safe = (
        (exg > 0.10)                         # green vegetation
        | (v_ch > 140)                       # bright
        | (texture > 10.0)                   # highly textured
        | ((s_ch > 80) & (v_ch > 60))       # saturated + not very dark
    )
    shadow &= ~safe

    # --- Global guard: skip if removing too much ---
    removing = int(np.sum(shadow))
    if removing > 0.30 * original_count:
        return mask

    refined = mask.copy()
    refined[shadow] = 0

    # Morphological cleanup
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k9, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k5, iterations=1)

    # Keep largest component
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined, connectivity=8
    )
    if n_comp > 1:
        lg = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        refined = np.where(labels == lg, 255, 0).astype(np.uint8)

    # Final guard: if result is too small vs original, shadow removal failed
    if np.sum(refined > 0) < 0.60 * original_count:
        return mask

    return refined
