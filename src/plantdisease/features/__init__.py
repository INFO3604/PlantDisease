"""
Feature extraction module for plant disease detection.

This module provides tools for extracting interpretable, traditional computer vision
features from preprocessed plant disease images. Three main feature groups are extracted:

1. **Texture Features (Gabor Filters)**: Filter bank responses with multiple orientations and frequencies
2. **Colour Features (CIELAB)**: Mean and standard deviation statistics for L, A, B channels
3. **Morphology Features**: Connected component analysis yielding lesion counts, area statistics, shape metrics, etc.

All features are computed on masked regions (disease/leaf pixels only) and returned as a flat dictionary
suitable for any traditional classifier (e.g., Random Forest, SVM, logistic regression).

For integration with the preprocessing pipeline::

    from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
    from plantdisease.features import extract_features_from_pipeline_result

    pipeline = PreprocessingPipeline()
    result = pipeline.run(image)
    features = extract_features_from_pipeline_result(result)

For batch processing with pandas::

    from plantdisease.features import extract_features_batch
    import pandas as pd

    df = extract_features_batch(images, leaf_masks, disease_masks)
    df.to_csv("features.csv")

"""

from .extract_features import (
    ColourFeatureExtractor,
    GaborTextureExtractor,
    MorphologyFeatureExtractor,
    extract_features,
    extract_features_batch,
    extract_features_from_pipeline_result,
    print_feature_summary,
)

__all__ = [
    "extract_features",
    "extract_features_batch",
    "extract_features_from_pipeline_result",
    "print_feature_summary",
    "GaborTextureExtractor",
    "ColourFeatureExtractor",
    "MorphologyFeatureExtractor",
]
