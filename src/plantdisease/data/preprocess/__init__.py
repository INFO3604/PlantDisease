"""Image preprocessing functions.

The primary preprocessing pipeline uses rembg background removal,
HSV shadow removal, and HSV disease segmentation.
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
from .background import remove_background_rembg
from .shadow import remove_shadows, remove_shadows_hsv_threshold
from .disease import DiseaseSegmenter, SeverityMetrics
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
    # Background removal
    "remove_background_rembg",
    # Shadow removal
    "remove_shadows",
    "remove_shadows_hsv_threshold",
    # Disease segmentation
    "DiseaseSegmenter",
    "SeverityMetrics",
    # Preprocessing pipeline
    "PreprocessingPipeline",
    "PipelineResult",
]
