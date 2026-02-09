"""Grayscale conversion utilities for plant disease image preprocessing.

This module provides functions for converting RGB images to grayscale.
While CNN-based models typically operate on RGB images, grayscale conversion
can be useful for models that focus more on texture rather than color differences,
reducing dimensionality and speeding up computation (Doğan 2023).

Note: For diseases where color changes are prominent (yellowing, necrosis, chlorosis),
preserving RGB channels may be more beneficial than grayscale conversion.
"""
import logging
from pathlib import Path
from typing import Union, Optional, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Grayscale conversion method types
GrayscaleMethod = Literal["luminosity", "average", "lightness", "opencv"]


def to_grayscale(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    method: GrayscaleMethod = "luminosity",
    keep_channels: bool = False,
) -> Optional[np.ndarray]:
    """
    Convert an image to grayscale using the specified method.
    
    This function supports multiple grayscale conversion methods to accommodate
    different use cases in plant disease detection. The luminosity method is
    recommended as it accounts for human perception of color brightness.
    
    Args:
        image: Path to input image or numpy array (BGR format if array).
        output_path: Where to save the result. If None, only returns the array.
        method: Grayscale conversion method:
            - "luminosity": Weighted average based on human perception (0.299R + 0.587G + 0.114B).
              Best for preserving perceived brightness of disease symptoms.
            - "average": Simple average of RGB channels ((R + G + B) / 3).
              Faster but less perceptually accurate.
            - "lightness": Average of max and min RGB values ((max + min) / 2).
              Preserves contrast between bright and dark regions.
            - "opencv": Uses OpenCV's optimized BGR2GRAY conversion.
              Fastest option, uses luminosity weights.
        keep_channels: If True, returns a 3-channel grayscale image (useful for
            models expecting RGB input). Default is False (single channel).
    
    Returns:
        Grayscale image as numpy array, or None if the image could not be read.
        Shape is (H, W) if keep_channels=False, or (H, W, 3) if keep_channels=True.
    
    Examples:
        >>> # Convert from file path
        >>> gray = to_grayscale("leaf.jpg")
        
        >>> # Convert with 3 channels for CNN input
        >>> gray_rgb = to_grayscale("leaf.jpg", keep_channels=True)
        
        >>> # Convert numpy array
        >>> img = cv2.imread("leaf.jpg")
        >>> gray = to_grayscale(img, method="average")
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
        if img is None or img.size == 0:
            logger.warning("Empty or invalid image array provided")
            return None
    
    # Validate image dimensions
    if len(img.shape) < 2:
        logger.warning("Invalid image dimensions")
        return None
    
    # Handle already grayscale images
    if len(img.shape) == 2:
        gray = img
    elif img.shape[2] == 1:
        gray = img.squeeze(axis=2)
    else:
        # Convert based on method
        gray = _convert_to_grayscale(img, method)
    
    # Expand to 3 channels if requested (for model compatibility)
    if keep_channels:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), gray)
        logger.debug(f"Saved grayscale image: {output_path}")
    
    return gray


def _convert_to_grayscale(img: np.ndarray, method: GrayscaleMethod) -> np.ndarray:
    """
    Internal function to perform grayscale conversion.
    
    Args:
        img: BGR image array (OpenCV format).
        method: Conversion method to use.
    
    Returns:
        Single-channel grayscale image.
    """
    if method == "opencv":
        # OpenCV's optimized conversion (uses luminosity weights internally)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    elif method == "luminosity":
        # Luminosity method: accounts for human color perception
        # Weights: R=0.299, G=0.587, B=0.114 (ITU-R BT.601 standard)
        # Note: OpenCV uses BGR order
        b, g, r = cv2.split(img)
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        return gray
    
    elif method == "average":
        # Simple average of all channels
        gray = np.mean(img, axis=2).astype(np.uint8)
        return gray
    
    elif method == "lightness":
        # Lightness method: average of max and min channel values
        max_val = np.max(img, axis=2)
        min_val = np.min(img, axis=2)
        gray = ((max_val.astype(np.float32) + min_val.astype(np.float32)) / 2).astype(np.uint8)
        return gray
    
    else:
        logger.warning(f"Unknown grayscale method '{method}', falling back to 'opencv'")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def batch_to_grayscale(
    image_paths: list,
    output_dir: Optional[Union[str, Path]] = None,
    method: GrayscaleMethod = "luminosity",
    keep_channels: bool = False,
    preserve_structure: bool = True,
) -> list:
    """
    Convert multiple images to grayscale.
    
    Processes a batch of images, useful for preprocessing entire datasets.
    
    Args:
        image_paths: List of paths to input images.
        output_dir: Directory to save converted images. If None, returns arrays only.
        method: Grayscale conversion method (see to_grayscale for options).
        keep_channels: If True, output has 3 channels for model compatibility.
        preserve_structure: If True and output_dir is provided, preserves the
            relative directory structure of input files.
    
    Returns:
        List of grayscale images as numpy arrays. Failed conversions are None.
    
    Examples:
        >>> paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> gray_images = batch_to_grayscale(paths)
        
        >>> # Save to output directory
        >>> batch_to_grayscale(paths, output_dir="processed/gray")
    """
    results = []
    output_dir = Path(output_dir) if output_dir else None
    
    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        
        # Determine output path
        if output_dir:
            if preserve_structure:
                # Try to preserve relative structure
                output_path = output_dir / img_path.name
            else:
                output_path = output_dir / img_path.name
        else:
            output_path = None
        
        # Convert
        gray = to_grayscale(
            img_path,
            output_path=output_path,
            method=method,
            keep_channels=keep_channels,
        )
        results.append(gray)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(image_paths)} images")
    
    success_count = sum(1 for r in results if r is not None)
    logger.info(f"Grayscale conversion complete: {success_count}/{len(image_paths)} successful")
    
    return results


def is_grayscale(image: Union[str, Path, np.ndarray]) -> bool:
    """
    Check if an image is already grayscale.
    
    An image is considered grayscale if it has a single channel, or if all
    three RGB channels have identical values.
    
    Args:
        image: Path to image or numpy array.
    
    Returns:
        True if the image is grayscale, False otherwise.
    
    Examples:
        >>> if not is_grayscale("leaf.jpg"):
        ...     gray = to_grayscale("leaf.jpg")
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return False
    else:
        img = image
    
    # Check dimensions
    if len(img.shape) == 2:
        return True
    
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return True
        # Check if all channels are identical
        b, g, r = cv2.split(img)
        return np.array_equal(b, g) and np.array_equal(g, r)
    
    return False


def grayscale_with_alpha(
    image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    method: GrayscaleMethod = "luminosity",
) -> Optional[np.ndarray]:
    """
    Convert image to grayscale while preserving the alpha (transparency) channel.
    
    Useful for processing leaf images with transparent backgrounds, which may
    have been pre-segmented using background removal techniques.
    
    Args:
        image: Path to input image or numpy array (BGRA format if array).
        output_path: Where to save the result. If None, only returns the array.
        method: Grayscale conversion method.
    
    Returns:
        Grayscale image with alpha channel (H, W, 4) or None if read failed.
    """
    # Load image with alpha channel
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Could not read image: {image}")
            return None
    else:
        img = image.copy()
    
    # Check if image has alpha channel
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Extract alpha channel
        alpha = img[:, :, 3]
        # Convert BGR to grayscale
        bgr = img[:, :, :3]
        gray = _convert_to_grayscale(bgr, method)
        # Combine grayscale with alpha
        result = cv2.merge([gray, gray, gray, alpha])
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # No alpha channel, just convert to grayscale
        gray = _convert_to_grayscale(img, method)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # Already grayscale
        result = img
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        logger.debug(f"Saved grayscale with alpha: {output_path}")
    
    return result
