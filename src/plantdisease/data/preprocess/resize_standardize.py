"""Image resizing and standardization."""
import logging
import cv2
import numpy as np
from pathlib import Path
from src.plantdisease import config

logger = logging.getLogger(__name__)

def resize_and_standardize(image_path, output_path=None, size=None, 
                           normalize=True, mean=None, std=None):
    """
    Resize image to target size and optionally normalize.
    
    Args:
        image_path: Path to input image
        output_path: Where to save (if None, returns array)
        size: Target size (default: config.IMG_SIZE)
        normalize: Whether to apply normalization
        mean: Normalization mean (default: config.NORMALIZE_MEAN)
        std: Normalization std (default: config.NORMALIZE_STD)
    
    Returns:
        Processed image array (if output_path is None)
    """
    if size is None:
        size = (config.IMG_SIZE, config.IMG_SIZE)
    if mean is None:
        mean = config.NORMALIZE_MEAN
    if std is None:
        std = config.NORMALIZE_STD
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    # Resize
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize (if requested)
    if normalize:
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    
    # Save or return
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save in float format or convert back to uint8
        if normalize:
            img = ((img * std) + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), img)
        logger.debug(f"Saved: {output_path}")
    
    return img
