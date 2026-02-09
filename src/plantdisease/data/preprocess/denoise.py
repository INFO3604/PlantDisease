"""Image denoising routines."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def denoise_median(image_path, kernel_size=5, output_path=None):
    """
    Apply median filter for denoising.
    
    Args:
        image_path: Path to input image
        kernel_size: Size of median filter (must be odd)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    denoised = cv2.medianBlur(img, kernel_size)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (median): {output_path}")
    
    return denoised

def denoise_bilateral(image_path, diameter=9, sigma_color=75, sigma_space=75, output_path=None):
    """
    Apply bilateral filter (edge-preserving denoising).
    
    Args:
        image_path: Path to input image
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    denoised = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (bilateral): {output_path}")
    
    return denoised

def denoise_gaussian(image_path, kernel_size=5, sigma=1.0, output_path=None):
    """
    Apply Gaussian blur for denoising.
    
    Args:
        image_path: Path to input image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image (if output_path is None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None
    
    denoised = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (Gaussian): {output_path}")
    
    return denoised
