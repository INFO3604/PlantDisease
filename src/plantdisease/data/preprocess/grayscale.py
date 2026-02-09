"""Grayscale conversion utilities."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def to_grayscale(image_path, output_path=None):
    """
    Convert image to grayscale.
    
    Args:
        image_path: Path to input image
        output_path: Where to save (if None, returns array)
    
    Returns:
        Grayscale image array (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), gray)
        logger.debug(f"Saved grayscale: {output_path}")
    
    return gray
