"""Contrast enhancement utilities."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def enhance_contrast_clahe(image_path, clip_limit=2.0, tile_size=8, output_path=None):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image_path: Path to input image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for local histogram equalization
        output_path: Where to save (if None, returns array)
    
    Returns:
        Enhanced image (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)
    
    lab[:, :, 0] = l_enhanced
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        logger.debug(f"Enhanced contrast (CLAHE): {output_path}")
    
    return enhanced

def enhance_contrast_histogram(image_path, output_path=None):
    """
    Global histogram equalization.
    
    Args:
        image_path: Path to input image
        output_path: Where to save (if None, returns array)
    
    Returns:
        Enhanced image (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    # Convert to LAB, equalize L channel, convert back
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    l_equalized = cv2.equalizeHist(l_channel)
    lab[:, :, 0] = l_equalized
    
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        logger.debug(f"Enhanced contrast (histogram): {output_path}")
    
    return enhanced
