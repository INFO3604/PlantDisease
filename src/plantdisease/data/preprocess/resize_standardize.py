"""
resize_standardize.py

Image resizing and standardization module for plant disease detection.
This module prepares images for model input by resizing them to a fixed
dimension and normalizing pixel intensity values.

"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Resize an image to a fixed target size.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB format.
    target_size : tuple(int, int)
        Desired image size (width, height).

    Returns
    -------
    np.ndarray
        Resized image.
    """
    if image is None:
        raise ValueError("Input image is None.")

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized


def standardize_image(
    image: np.ndarray,
    use_z_score: bool = False,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Standardize image pixel values.

    Two modes are supported:
    1. Min-max normalization to [0, 1]
    2. Z-score normalization (optional)

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB).
    use_z_score : bool
        Whether to apply z-score normalization.
    mean : np.ndarray, optional
        Channel-wise mean (used only for z-score).
    std : np.ndarray, optional
        Channel-wise standard deviation (used only for z-score).

    Returns
    -------
    np.ndarray
        Standardized image.
    """
    image = image.astype(np.float32)

    # Min-max normalization
    image /= 255.0

    if use_z_score:
        if mean is None or std is None:
            raise ValueError(
                "Mean and standard deviation must be provided for z-score normalization."
            )

        image = (image - mean) / std

    return image


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    use_z_score: bool = False,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Load, resize, and standardize a single image.

    Parameters
    ----------
    image_path : str
        Path to input image.
    target_size : tuple(int, int)
        Desired image size.
    use_z_score : bool
        Whether to apply z-score normalization.
    mean : np.ndarray, optional
        Channel-wise mean.
    std : np.ndarray, optional
        Channel-wise standard deviation.

    Returns
    -------
    np.ndarray
        Preprocessed image ready for model input.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image (BGR → RGB)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize_image(image, target_size)
    image = standardize_image(image, use_z_score, mean, std)

    return image


def preprocess_directory(
    input_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    use_z_score: bool = False,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Preprocess all images in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing images.
    target_size : tuple(int, int)
        Desired image size.
    use_z_score : bool
        Whether to apply z-score normalization.
    mean : np.ndarray, optional
        Channel-wise mean.
    std : np.ndarray, optional
        Channel-wise standard deviation.

    Returns
    -------
    np.ndarray
        Array of preprocessed images.
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Invalid directory: {input_dir}")

    images = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, filename)
            processed = preprocess_image(
                image_path,
                target_size,
                use_z_score,
                mean,
                std
            )
            images.append(processed)

    if len(images) == 0:
        raise ValueError("No valid images found in directory.")

    return np.array(images)
