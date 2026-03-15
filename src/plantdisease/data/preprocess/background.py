"""Background removal and segmentation."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def remove_background_rembg(image: np.ndarray) -> np.ndarray:
    """Remove background using the rembg deep learning model.

    Converts the input BGR image to RGBA, runs rembg to make the
    background transparent, and returns the RGBA result as a NumPy array.

    Args:
        image: Input BGR image (uint8 NumPy array).

    Returns:
        RGBA image (uint8 NumPy array) with transparent background.
    """
    from rembg import remove
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    result = remove(pil_img)
    return np.array(result)
