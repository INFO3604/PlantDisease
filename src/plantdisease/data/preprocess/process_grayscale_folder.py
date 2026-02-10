"""Region of Interest (ROI) extraction and cropping."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def extract_roi(image_path, roi_coords, output_path=None):
    """
    Extract ROI from image given coordinates (x, y, w, h).
    
    Args:
        image_path: Path to input image
        roi_coords: Tuple (x, y, width, height)
        output_path: Where to save (if None, returns array)
    
    Returns:
        ROI image array (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    x, y, w, h = roi_coords
    roi = img[y:y+h, x:x+w]
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), roi)
        logger.debug(f"Extracted ROI: {output_path}")
    
    return roi

def crop_center(image_path, crop_fraction=0.9, output_path=None):
    """
    Crop center portion of image.
    
    Args:
        image_path: Path to input image
        crop_fraction: Fraction of image to keep (0-1)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Cropped image array (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    crop_h = int(h * crop_fraction)
    crop_w = int(w * crop_fraction)
    
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    
    cropped = img[y:y+crop_h, x:x+crop_w]
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)
        logger.debug(f"Cropped center: {output_path}")
    
    return cropped