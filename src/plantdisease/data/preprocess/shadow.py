"""
Shadow detection and suppression module for plant disease images.

This module handles shadow detection and removal in field images using
HSV color space analysis. Shadows are identified by low Value (brightness)
regions and can be attenuated or removed.
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
    
    Uses HSV color space to identify low-value (dark) regions
    that correspond to shadows in field images.
    """
    
    def __init__(
        self,
        v_threshold: int = 50,
        attenuation_factor: float = 0.5,
        min_shadow_area: int = 100
    ):
        """
        Initialize shadow handler.
        
        Args:
            v_threshold: HSV Value below this is considered shadow
            attenuation_factor: How much to attenuate shadows (0=remove, 1=keep)
            min_shadow_area: Minimum contour area to consider as shadow
        """
        self.v_threshold = v_threshold
        self.attenuation_factor = attenuation_factor
        self.min_shadow_area = min_shadow_area
    
    def detect_shadow_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions in image using HSV Value channel.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Binary mask where 255 = shadow region
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract Value channel
        v_channel = hsv[:, :, 2]
        
        # Threshold to find dark (shadow) regions
        shadow_mask = np.where(v_channel < self.v_threshold, 255, 0).astype(np.uint8)
        
        # Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small contours
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(shadow_mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_shadow_area:
                cv2.drawContours(cleaned_mask, [contour], -1, 255, -1)
        
        return cleaned_mask
    
    def suppress_shadows(
        self,
        image: np.ndarray,
        shadow_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Suppress (attenuate) shadow regions in image.
        
        Args:
            image: Input image in BGR format
            shadow_mask: Pre-computed shadow mask (optional)
            
        Returns:
            Image with shadows attenuated
        """
        if shadow_mask is None:
            shadow_mask = self.detect_shadow_mask(image)
        
        # Convert to float for processing
        result = image.astype(np.float32)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Create shadow region mask (normalized)
        shadow_regions = shadow_mask > 0
        
        if np.any(shadow_regions):
            # Boost the V channel in shadow regions
            # Calculate how much to boost based on current value
            v_channel = hsv[:, :, 2]
            
            # Increase brightness in shadow regions
            boost_factor = 1.0 + (1.0 - self.attenuation_factor)
            v_channel[shadow_regions] = np.clip(
                v_channel[shadow_regions] * boost_factor,
                0, 255
            )
            
            hsv[:, :, 2] = v_channel
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result.astype(np.uint8)
    
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
