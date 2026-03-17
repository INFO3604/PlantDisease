"""Image preprocessing functions.

The primary preprocessing pipeline uses rembg background removal,
HSV shadow removal, and HSV disease segmentation.
"""
from .resize_standardize import (
    resize_image,
    standardize_image,
    preprocess_image,
    preprocess_directory,
)
from .background import remove_background_rembg
from .shadow import remove_shadows, remove_shadows_hsv_threshold
from .disease import DiseaseSegmenter, SeverityMetrics
from .pipeline import PreprocessingPipeline, PipelineResult

__all__ = [
    # Resize/Standardize
    "resize_image",
    "standardize_image",
    "preprocess_image",
    "preprocess_directory",
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
