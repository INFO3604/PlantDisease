"""Image preprocessing functions.

The primary preprocessing pipeline is PreprocessingPipeline which uses
<<<<<<< HEAD
SAM-first full-leaf isolation on AGCWD output and watershed disease
segmentation constrained to the leaf mask.
=======
LAB a*-channel segmentation (no GrabCut) and Mahalanobis disease
detection on the segmented leaf to avoid shadow false-positives.
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
"""
from .grayscale import (
    to_grayscale,
    batch_to_grayscale,
    is_grayscale,
    grayscale_with_alpha,
)
from .resize_standardize import (
    resize_image,
    standardize_image,
    preprocess_image,
    preprocess_directory,
)
from .denoise import denoise_median, denoise_bilateral, denoise_gaussian
<<<<<<< HEAD
from .leaf_segmentation import segment_leaf, apply_mask
=======
from .leaf_segmentation import (
    ColorIndexSegmenter,
    LABSegmenter,
    SLICSegmenter,
    SegmentationMethod,
)
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
from .pipeline import PreprocessingPipeline, PipelineResult

__all__ = [
    # Grayscale
    "to_grayscale",
    "batch_to_grayscale",
    "is_grayscale",
    "grayscale_with_alpha",
    # Resize/Standardize
    "resize_image",
    "standardize_image",
    "preprocess_image",
    "preprocess_directory",
    # Denoise
    "denoise_median",
    "denoise_bilateral",
    "denoise_gaussian",
<<<<<<< HEAD
    # Leaf segmentation helpers
    "segment_leaf",
    "apply_mask",
=======
    # Leaf segmentation (non-GrabCut)
    "ColorIndexSegmenter",
    "LABSegmenter",
    "SLICSegmenter",
    "SegmentationMethod",
>>>>>>> 03c98b45fbf4486ecdada1bf40e1c6e21ec31f36
    # Preprocessing pipeline (primary — recommended)
    "PreprocessingPipeline",
    "PipelineResult",
]
