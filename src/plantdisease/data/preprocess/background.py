"""Background removal and segmentation."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def remove_background_hsv(image_path, output_path=None, lower_green=None, upper_green=None):
    """
    Remove background using HSV color space (for green leaves).
    
    Args:
        image_path: Path to input image
        output_path: Where to save (if None, returns array)
        lower_green: Lower HSV bound for green
        upper_green: Upper HSV bound for green
    
    Returns:
        Background-removed image (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    if lower_green is None:
        lower_green = np.array([25, 40, 40])
    if upper_green is None:
        upper_green = np.array([95, 255, 255])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img, img, mask=mask)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        logger.debug(f"Removed background: {output_path}")
    
    return result

def get_foreground_mask(image_path):
    """
    Get binary mask of foreground (plant) areas.
    
    Args:
        image_path: Path to input image
    
    Returns:
        Binary mask
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    # Simple threshold based on RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return mask
