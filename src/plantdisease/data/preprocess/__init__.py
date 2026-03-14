"""Image preprocessing functions.

The primary preprocessing pipeline is PreprocessingPipeline which uses
SAM-first full-leaf isolation on AGCWD output and watershed disease
segmentation constrained to the leaf mask.
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
from .leaf_segmentation import segment_leaf, apply_mask
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
    # Leaf segmentation helpers
    "segment_leaf",
    "apply_mask",
    # Preprocessing pipeline (primary — recommended)
    "PreprocessingPipeline",
    "PipelineResult",
]
