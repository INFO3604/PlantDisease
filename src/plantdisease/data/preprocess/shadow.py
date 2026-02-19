"""
Shadow detection and suppression module for plant disease images.

This module handles shadow detection and removal in field images using
HSV color space analysis. Shadows are identified by low Value (brightness)
AND low Saturation regions, distinguishing them from actual dark disease spots.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ShadowHandler:
    """
    Shadow detection and suppression for leaf images.
    
    Uses HSV color space to identify shadow regions characterized by:
    - Low Value (brightness)
    - Low Saturation (shadows desaturate colors)
    - Smooth gradients (shadows have soft edges unlike disease spots)
    """
    
    def __init__(
        self,
        v_threshold: int = 80,
        s_threshold: int = 60,
        attenuation_factor: float = 0.3,
        min_shadow_area: int = 200,
        use_adaptive: bool = True
    ):
        """
        Initialize shadow handler.
        
        Args:
            v_threshold: HSV Value below this is considered potentially shadow
            s_threshold: HSV Saturation below this (combined with low V) = shadow
            attenuation_factor: How much to attenuate shadows (0=remove, 1=keep)
            min_shadow_area: Minimum contour area to consider as shadow
            use_adaptive: Use adaptive thresholds based on image statistics
        """
        self.v_threshold = v_threshold
        self.s_threshold = s_threshold
        self.attenuation_factor = attenuation_factor
        self.min_shadow_area = min_shadow_area
        self.use_adaptive = use_adaptive
    
    def detect_shadow_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions using multiple criteria.
        
        Shadows are characterized by:
        1. Low brightness (V channel)
        2. Low saturation (S channel) - shadows desaturate colors
        3. Gradual transitions (unlike sharp disease borders)
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Binary mask where 255 = shadow region
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Adaptive thresholds based on image statistics
        if self.use_adaptive:
            v_mean = np.mean(v_channel)
            v_std = np.std(v_channel)
            s_mean = np.mean(s_channel)
            
            # Shadow threshold: below mean - 0.5*std for V
            v_thresh = max(40, min(self.v_threshold, v_mean - 0.5 * v_std))
            # For saturation: shadows have low saturation
            s_thresh = max(40, min(self.s_threshold, s_mean - 0.3 * np.std(s_channel)))
        else:
            v_thresh = self.v_threshold
            s_thresh = self.s_threshold
        
        # Primary shadow detection: Low V AND Low S
        # Shadows desaturate AND darken the image
        low_v_mask = v_channel < v_thresh
        low_s_mask = s_channel < s_thresh
        shadow_mask_primary = (low_v_mask & low_s_mask).astype(np.uint8) * 255
        
        # Secondary: Very dark regions (V < 30) regardless of saturation
        # These are likely deep shadows or black backgrounds
        very_dark_mask = (v_channel < 30).astype(np.uint8) * 255
        
        # Combine masks
        shadow_mask = cv2.bitwise_or(shadow_mask_primary, very_dark_mask)
        
        # Exclude regions with high saturation and specific hues (disease colors)
        # Brown disease: H=0-25, S>80, V=30-150
        # Yellow disease: H=15-35, S>80
        disease_like = (
            (s_channel > 80) & 
            (((h_channel >= 0) & (h_channel <= 35)) | ((h_channel >= 160) & (h_channel <= 180)))
        ).astype(np.uint8) * 255
        
        # Remove disease-like regions from shadow mask
        shadow_mask = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(disease_like))
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep only larger shadow regions (small dark spots might be disease)
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(shadow_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_shadow_area:
                # Additional check: shadows tend to have smooth, elongated shapes
                # Disease spots tend to be more circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Shadows often have low circularity (elongated)
                    # But we'll keep it if it's a large area regardless
                    if circularity < 0.7 or area > 1000:
                        cv2.drawContours(cleaned_mask, [contour], -1, 255, -1)
        
        return cleaned_mask
    
    def suppress_shadows(
        self,
        image: np.ndarray,
        shadow_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Suppress shadow regions using illumination normalization.
        
        Uses LAB color space for better results - normalizes L channel
        while preserving color information.
        
        Args:
            image: Input image in BGR format
            shadow_mask: Pre-computed shadow mask (optional)
            
        Returns:
            Image with shadows suppressed
        """
        if shadow_mask is None:
            shadow_mask = self.detect_shadow_mask(image)
        
        # Use LAB color space for illumination normalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Estimate illumination using large bilateral filter
        illumination = cv2.bilateralFilter(l_channel, 15, 75, 75)
        
        # Calculate target illumination (mean of non-shadow regions)
        non_shadow_mask = shadow_mask == 0
        if np.any(non_shadow_mask):
            target_illumination = np.mean(l_channel[non_shadow_mask])
        else:
            target_illumination = np.mean(l_channel)
        
        # Normalize illumination
        normalized_l = np.clip(
            l_channel * (target_illumination / (illumination + 1e-6)),
            0, 255
        )
        
        # Apply stronger correction in shadow regions
        shadow_regions = shadow_mask > 0
        if np.any(shadow_regions):
            # Boost shadow regions more aggressively
            boost = 1.0 + (1.0 - self.attenuation_factor) * 0.5
            normalized_l[shadow_regions] = np.clip(
                normalized_l[shadow_regions] * boost,
                0, 255
            )
        
        lab[:, :, 0] = normalized_l.astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def remove_shadows_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows using Multi-Scale Retinex (MSR).
        
        Retinex separates illumination from reflectance, effectively
        removing shadows while preserving color.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Shadow-removed image
        """
        # Convert to float
        img_float = image.astype(np.float32) + 1.0  # Avoid log(0)
        
        # Multi-scale Retinex
        scales = [15, 80, 250]
        retinex = np.zeros_like(img_float)
        
        for scale in scales:
            blur = cv2.GaussianBlur(img_float, (0, 0), scale)
            retinex += np.log10(img_float) - np.log10(blur + 1.0)
        
        retinex = retinex / len(scales)
        
        # Normalize to 0-255
        for i in range(3):
            channel = retinex[:, :, i]
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                retinex[:, :, i] = (channel - min_val) / (max_val - min_val) * 255
            else:
                retinex[:, :, i] = 127
        
        return np.clip(retinex, 0, 255).astype(np.uint8)
    
    def remove_shadows_illumination_normalization(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Remove shadows using illumination normalization.
        
        This method uses LAB color space and normalizes the L channel
        to reduce shadow effects.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Image with shadows reduced
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Apply bilateral filter to estimate illumination
        illumination = cv2.bilateralFilter(l_channel, 15, 80, 80)
        
        # Normalize
        mean_illumination = np.mean(illumination)
        normalized_l = np.clip(
            l_channel * (mean_illumination / (illumination + 1e-6)),
            0, 255
        ).astype(np.uint8)
        
        lab[:, :, 0] = normalized_l
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def get_shadow_stats(self, image: np.ndarray) -> Dict:
        """
        Compute shadow statistics for an image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary with shadow statistics
        """
        shadow_mask = self.detect_shadow_mask(image)
        
        total_pixels = image.shape[0] * image.shape[1]
        shadow_pixels = np.sum(shadow_mask > 0)
        shadow_ratio = shadow_pixels / total_pixels
        
        return {
            'shadow_pixels': int(shadow_pixels),
            'total_pixels': int(total_pixels),
            'shadow_ratio': float(shadow_ratio),
            'has_significant_shadows': shadow_ratio > 0.1
        }


def suppress_shadows(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    v_threshold: int = 50,
    attenuation_factor: float = 0.5
) -> np.ndarray:
    """
    Suppress shadows in an image file.
    
    Args:
        image_path: Path to input image
        output_path: Path to save result (optional)
        v_threshold: HSV Value threshold for shadow detection
        attenuation_factor: Shadow attenuation factor
        
    Returns:
        Shadow-suppressed image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    handler = ShadowHandler(
        v_threshold=v_threshold,
        attenuation_factor=attenuation_factor
    )
    
    result = handler.suppress_shadows(image)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        logger.debug(f"Saved shadow-suppressed image: {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Shadow detection and suppression")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--threshold", type=int, default=50, help="V threshold")
    parser.add_argument("--attenuation", type=float, default=0.5, help="Attenuation factor")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    result = suppress_shadows(
        args.input,
        args.output,
        args.threshold,
        args.attenuation
    )
    
    print(f"Processed image shape: {result.shape}")
