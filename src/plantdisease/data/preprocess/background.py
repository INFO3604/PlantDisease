"""Background removal and segmentation."""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def remove_background_rembg(image: np.ndarray, session=None) -> np.ndarray:
    """Remove background using the rembg deep learning model.

    Converts the input BGR image to RGBA, runs rembg to make the
    background transparent, and returns the RGBA result as a NumPy array.

    Args:
        image: Input BGR image (uint8 NumPy array).
        session: Optional pre-created rembg session for reuse.

    Returns:
        RGBA image (uint8 NumPy array) with transparent background.
    """
    from rembg import remove, new_session
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    result = remove(pil_img, session=session)
    return np.array(result)


# Cached singleton session — loaded once on first use
_default_session = None


def get_rembg_session():
    """Return a cached rembg inference session (singleton)."""
    global _default_session
    if _default_session is None:
        from rembg import new_session
        _default_session = new_session("u2net")
    return _default_session
