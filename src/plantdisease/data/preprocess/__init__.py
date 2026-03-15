"""Image preprocessing functions.

The primary preprocessing pipeline uses DeepLabV3+ leaf segmentation
(with GrabCut refinement and multi-strategy shadow removal) and
multi-scale U-Net-inspired disease detection.
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
    segment_leaf,
    apply_mask,
    WatershedSegmenter,
    DeepLabV3Segmenter,
    SegmentationMethod,
    SegmentationResult,
)
from .shadow import remove_shadows
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
    # Leaf segmentation
    "segment_leaf",
    "apply_mask",
    "WatershedSegmenter",
    "DeepLabV3Segmenter",
    "SegmentationMethod",
    "SegmentationResult",
    # Shadow removal
    "remove_shadows",
    # Preprocessing pipeline
    "PreprocessingPipeline",
    "PipelineResult",
]
