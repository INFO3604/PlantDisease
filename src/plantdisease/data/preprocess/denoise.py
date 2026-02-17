"""
Image denoising routines for plant disease preprocessing.

This module provides multiple denoising methods:
- Median filter: Good for salt-and-pepper noise
- Gaussian blur: Good for general smoothing
- Bilateral filter: Edge-preserving denoising
- Non-local means: Best quality but slowest

All methods support both file paths and numpy arrays as input.
"""
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def denoise_median(
    image: Union[str, Path, np.ndarray],
    kernel_size: int = 5,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Apply median filter for denoising.
    
    Median filtering is particularly effective for removing salt-and-pepper
    noise while preserving edges. Good choice for images with impulse noise.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        kernel_size: Size of median filter (must be odd)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image array, or None if loading failed
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    denoised = cv2.medianBlur(img, kernel_size)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (median): {output_path}")
    
    return denoised


def denoise_bilateral(
    image: Union[str, Path, np.ndarray],
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Apply bilateral filter (edge-preserving denoising).
    
    Bilateral filtering smooths images while keeping edges sharp.
    Useful for reducing noise without blurring important features
    like disease lesion boundaries.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image array, or None if loading failed
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    denoised = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (bilateral): {output_path}")
    
    return denoised


def denoise_gaussian(
    image: Union[str, Path, np.ndarray],
    kernel_size: int = 5,
    sigma: float = 1.0,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Apply Gaussian blur for denoising.
    
    Gaussian blur reduces high-frequency noise but also blurs edges.
    Simple and fast, suitable when some edge blurring is acceptable.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image array, or None if loading failed
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    denoised = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (Gaussian): {output_path}")
    
    return denoised


def denoise_nlm(
    image: Union[str, Path, np.ndarray],
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
    output_path: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Apply Non-Local Means denoising.
    
    NLM provides superior denoising quality by averaging pixels based
    on similarity of their neighborhoods. Slower but best results for
    photographic noise.
    
    Args:
        image: Path to input image or numpy array (BGR format)
        h: Filter strength. Higher removes more noise but removes detail
        template_window_size: Size of template patch (odd)
        search_window_size: Size of area to search for similar patches (odd)
        output_path: Where to save (if None, returns array)
    
    Returns:
        Denoised image array, or None if loading failed
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Use color version for BGR images
    if len(img.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h, h,
            template_window_size, search_window_size
        )
    else:
        denoised = cv2.fastNlMeansDenoising(
            img, None, h,
            template_window_size, search_window_size
        )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)
        logger.debug(f"Denoised (NLM): {output_path}")
    
    return denoised


class ImageDenoiser:
    """
    Configurable image denoiser supporting multiple methods.
    """
    
    METHODS = {
        'median': denoise_median,
        'bilateral': denoise_bilateral,
        'gaussian': denoise_gaussian,
        'nlm': denoise_nlm
    }
    
    def __init__(
        self,
        method: str = 'median',
        **kwargs
    ):
        """
        Initialize denoiser with specified method and parameters.
        
        Args:
            method: Denoising method ('median', 'bilateral', 'gaussian', 'nlm')
            **kwargs: Method-specific parameters
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS.keys())}")
        
        self.method = method
        self.params = kwargs
    
    def denoise(
        self,
        image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[np.ndarray]:
        """
        Apply configured denoising method.
        
        Args:
            image: Input image (path or array)
            output_path: Optional output path
            
        Returns:
            Denoised image array
        """
        func = self.METHODS[self.method]
        return func(image, output_path=output_path, **self.params)
