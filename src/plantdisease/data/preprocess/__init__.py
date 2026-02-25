"""Image preprocessing functions.

The primary preprocessing pipeline is PreprocessingPipeline which uses
LAB a*-channel segmentation (no GrabCut) and Mahalanobis disease
detection on the segmented leaf to avoid shadow false-positives.
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
from .leaf_segmentation import (
    ColorIndexSegmenter,
    LABSegmenter,
    SLICSegmenter,
    SegmentationMethod,
)
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
    # Leaf segmentation (non-GrabCut)
    "ColorIndexSegmenter",
    "LABSegmenter",
    "SLICSegmenter",
    "SegmentationMethod",
    # Preprocessing pipeline (primary — recommended)
    "PreprocessingPipeline",
    "PipelineResult",
]
