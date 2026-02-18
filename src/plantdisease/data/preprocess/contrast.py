"""
Contrast enhancement utilities for plant disease image preprocessing.

This module provides:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Global histogram equalization
- Adaptive gamma correction

Contrast enhancement helps reveal disease symptoms that may be 
obscured by poor lighting or low image contrast.
"""
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def enhance_contrast_clahe(
    image: Union[str, Path, np.ndarray],
    clip_limit: float = 2.0,
    tile_size: int = 8,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE operates on small tiles and limits contrast amplification to avoid
    noise amplification. Particularly useful for images with varying illumination.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for local histogram equalization
        output_path: Where to save (if None, returns array)
    
    Returns:
        Enhanced image (if output_path is None)
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)
    
    lab[:, :, 0] = l_enhanced
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        logger.debug(f"Enhanced contrast (CLAHE): {output_path}")
    
    return enhanced


def enhance_contrast_histogram(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Global histogram equalization.
    
    Spreads out the intensity values across the full range.
    Simple but effective for images with concentrated intensity distribution.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Enhanced image (if output_path is None)
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Convert to LAB, equalize L channel, convert back
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    l_equalized = cv2.equalizeHist(l_channel)
    lab[:, :, 0] = l_equalized
    
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        logger.debug(f"Enhanced contrast (histogram): {output_path}")
    
    return enhanced


def enhance_contrast_gamma(
    image: Union[str, Path, np.ndarray],
    gamma: float = 1.0,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Apply gamma correction for contrast adjustment.
    
    Gamma < 1.0 brightens the image (useful for underexposed images)
    Gamma > 1.0 darkens the image
    
    Args:
        image: Path to input image or numpy array (BGR format)
        gamma: Gamma value (1.0 = no change)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Enhanced image
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)
    
    enhanced = cv2.LUT(img, table)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        logger.debug(f"Enhanced contrast (gamma={gamma}): {output_path}")
    
    return enhanced


def auto_gamma_correction(
    image: Union[str, Path, np.ndarray],
    target_mean: float = 127.0,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Automatically adjust gamma to achieve target mean brightness.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        target_mean: Target mean brightness (0-255)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Enhanced image with auto-computed gamma
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Calculate current mean
    current_mean = np.mean(img)
    
    # Estimate gamma needed
    if current_mean > 0:
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0 + 1e-6)
        gamma = np.clip(gamma, 0.5, 2.5)  # Limit gamma range
    else:
        gamma = 1.0
    
    logger.debug(f"Auto gamma: current_mean={current_mean:.1f}, gamma={gamma:.2f}")
    
    return enhance_contrast_gamma(img, gamma, output_path)


class ContrastEnhancer:
    """
    Configurable contrast enhancer supporting multiple methods.
    """
    
    METHODS = {
        'clahe': enhance_contrast_clahe,
        'histogram': enhance_contrast_histogram,
        'gamma': enhance_contrast_gamma,
        'auto_gamma': auto_gamma_correction
    }
    
    def __init__(
        self,
        method: str = 'clahe',
        **kwargs
    ):
        """
        Initialize enhancer with specified method and parameters.
        
        Args:
            method: Enhancement method ('clahe', 'histogram', 'gamma', 'auto_gamma')
            **kwargs: Method-specific parameters
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS.keys())}")
        
        self.method = method
        self.params = kwargs
    
    def enhance(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[np.ndarray]:
        """
        Apply configured enhancement method.
        
        Args:
            image: Input image (path or array)
            output_path: Optional output path
            
        Returns:
            Enhanced image array
        """
        func = self.METHODS[self.method]
        return func(image, output_path=output_path, **self.params)
