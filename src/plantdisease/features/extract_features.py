"""
Feature extraction module for plant disease detection.

Extracts interpretable, traditional computer vision features from disease images:
  - Texture features: Gabor filter responses from disease regions
  - Colour features: CIELAB colour statistics (mean, std) + pixel ratios
  - Morphology features: Connected component analysis on disease masks

Features are returned as a flat dictionary suitable for any traditional classifier.
All statistics computed on masked regions (disease/leaf pixels only).
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from skimage.measure import regionprops, label

logger = logging.getLogger(__name__)


# =============================================================================
# Texture Feature Extraction (Gabor Filters)
# =============================================================================

class GaborTextureExtractor:
    """Extract Gabor filter-based texture features from disease regions.

    Applies a bank of Gabor filters with multiple orientations and frequencies
    to the disease region. Extracts statistical descriptors from filter
    responses: mean, standard deviation, and energy.

    Parameters
    ----------
    frequencies : list or tuple
        Spatial frequencies for Gabor filters (default: (0.05, 0.1, 0.2)).
    orientations : int
        Number of orientations to use (default: 4). Creates filters at
        angles: 0, π/4, π/2, 3π/4 (or similar based on orientations).
    sigma : float
        Standard deviation of Gaussian envelope (default: 3.0).
    kernel_size : int
        Size of the Gabor kernel (default: 21).
    """

    def __init__(
        self,
        frequencies: Tuple = (0.05, 0.1, 0.2),
        orientations: int = 4,
        sigma: float = 3.0,
        kernel_size: int = 21,
    ):
        self.frequencies = frequencies
        self.orientations = orientations
        self.sigma = sigma
        self.kernel_size = kernel_size
        self._filters = self._create_gabor_filters()

    def _create_gabor_filters(self) -> list:
        """Create a bank of Gabor filters.

        Returns
        -------
        list
            List of Gabor filter kernels.
        """
        filters = []
        for freq in self.frequencies:
            for orient in range(self.orientations):
                angle = (np.pi * orient) / self.orientations
                kernel = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    self.sigma,
                    angle,
                    1.0 / freq,
                    0.5,
                    0
                )
                filters.append(kernel)
        return filters

    def extract(
        self,
        image: np.ndarray,
        disease_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract Gabor texture features from disease region.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or RGB).
        disease_mask : np.ndarray
            Binary mask of disease regions (0 or 255).

        Returns
        -------
        dict
            Dictionary of Gabor feature values (mean, std, energy for each filter).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy().astype(np.uint8)

        # Normalize grayscale
        gray = gray.astype(np.float32) / 255.0

        # Ensure mask is binary
        mask_binary = (disease_mask > 127).astype(np.uint8)

        features = {}

        # Handle empty mask
        if mask_binary.sum() == 0:
            logger.warning("Empty disease mask provided for Gabor extraction")
            n_filters = len(self._filters)
            for i in range(n_filters):
                features[f"gabor_mean_{i}"] = 0.0
                features[f"gabor_std_{i}"] = 0.0
                features[f"gabor_energy_{i}"] = 0.0
            return features

        # Apply each Gabor filter
        for i, kernel in enumerate(self._filters):
            # Convolve image with Gabor filter
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)

            # Extract values only from disease region
            response_masked = response[mask_binary > 0]

            # Compute statistics
            mean_val = float(np.mean(response_masked))
            std_val = float(np.std(response_masked))
            energy_val = float(np.sum(response_masked ** 2) / max(len(response_masked), 1))

            features[f"gabor_mean_{i}"] = mean_val
            features[f"gabor_std_{i}"] = std_val
            features[f"gabor_energy_{i}"] = energy_val

        return features


# =============================================================================
# Colour Feature Extraction (CIELAB)
# =============================================================================

class ColourFeatureExtractor:
    """Extract CIELAB colour statistics from disease and leaf regions.

    Computes mean and standard deviation of lightness, a-component,
    b-component (LAB) for disease regions. Also computes pixel ratios
    (yellow, brown, disease) from the preprocessing pipeline.

    Parameters
    ----------
    compute_leaf_stats : bool
        If True, also compute LAB stats for the whole leaf region.
    """

    def __init__(self, compute_leaf_stats: bool = False):
        self.compute_leaf_stats = compute_leaf_stats

    def extract(
        self,
        image: np.ndarray,
        disease_mask: np.ndarray,
        yellow_mask: Optional[np.ndarray] = None,
        brown_mask: Optional[np.ndarray] = None,
        leaf_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract colour features from disease and leaf regions.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR).
        disease_mask : np.ndarray
            Binary mask of combined disease regions (0 or 255).
        yellow_mask : np.ndarray, optional
            Binary mask of yellow/chlorosis regions (0 or 255).
        brown_mask : np.ndarray, optional
            Binary mask of brown/necrosis regions (0 or 255).
        leaf_mask : np.ndarray, optional
            Binary mask of leaf region (0 or 255).

        Returns
        -------
        dict
            Dictionary of colour features (LAB mean/std + ratios).
        """
        features = {}

        # Compute pixel ratios
        total_leaf = (leaf_mask > 127).sum() if leaf_mask is not None else 1
        disease_ratio = (disease_mask > 127).sum() / max(total_leaf, 1)
        features["disease_ratio"] = float(disease_ratio)

        if yellow_mask is not None:
            yellow_ratio = (yellow_mask > 127).sum() / max(total_leaf, 1)
            features["yellow_ratio"] = float(yellow_ratio)

        if brown_mask is not None:
            brown_ratio = (brown_mask > 127).sum() / max(total_leaf, 1)
            features["brown_ratio"] = float(brown_ratio)

        # Ensure mask is binary
        disease_binary = (disease_mask > 127).astype(np.uint8)

        if disease_binary.sum() == 0:
            logger.warning("Empty disease mask for colour feature extraction")
            # Return default values (LAB only, no HSV)
            for suffix in ["_mean", "_std"]:
                for c in ["l", "a", "b"]:
                    features[f"disease_{c}{suffix}"] = 0.0
            if self.compute_leaf_stats and leaf_mask is not None:
                leaf_binary = (leaf_mask > 127).astype(np.uint8)
                if leaf_binary.sum() == 0:
                    for suffix in ["_mean", "_std"]:
                        for c in ["l", "a", "b"]:
                            features[f"leaf_{c}{suffix}"] = 0.0
            return features

        # Extract disease region statistics
        disease_features = self._extract_region_stats(
            image, disease_binary, prefix="disease_"
        )
        features.update(disease_features)

        # Extract leaf region statistics (if requested and mask provided)
        if self.compute_leaf_stats and leaf_mask is not None:
            leaf_binary = (leaf_mask > 127).astype(np.uint8)
            if leaf_binary.sum() > 0:
                leaf_features = self._extract_region_stats(
                    image, leaf_binary, prefix="leaf_"
                )
                features.update(leaf_features)

        return features

    def _extract_region_stats(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Extract CIELAB statistics for a masked region.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        mask : np.ndarray
            Binary mask.
        prefix : str
            Prefix for feature names.

        Returns
        -------
        dict
            LAB statistics only (Lightness, a-component, b-component).
        """
        features = {}

        # Convert to CIELAB colour space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Extract masked values
        l_vals = l[mask > 0]
        a_vals = a[mask > 0]
        b_vals = b[mask > 0]

        # LAB statistics only (removed HSV)
        features[f"{prefix}l_mean"] = float(np.mean(l_vals))
        features[f"{prefix}l_std"] = float(np.std(l_vals))
        features[f"{prefix}a_mean"] = float(np.mean(a_vals))
        features[f"{prefix}a_std"] = float(np.std(a_vals))
        features[f"{prefix}b_mean"] = float(np.mean(b_vals))
        features[f"{prefix}b_std"] = float(np.std(b_vals))

        return features


# =============================================================================
# Morphology Feature Extraction
# =============================================================================

class MorphologyFeatureExtractor:
    """Extract morphological features from disease masks using connected components.

    Computes lesion count, area statistics, perimeter, circularity,
    eccentricity, solidity, and extent from the binary disease mask.
    """

    def extract(
        self,
        disease_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract morphological features from disease regions.

        Parameters
        ----------
        disease_mask : np.ndarray
            Binary mask of disease regions (0 or 255).

        Returns
        -------
        dict
            Dictionary of morphology features.
        """
        features = {}

        # Ensure mask is binary
        mask_binary = (disease_mask > 127).astype(np.uint8)

        # Label connected components
        labeled_array, num_components = ndimage.label(mask_binary)

        # Handle empty mask
        if num_components == 0:
            logger.warning("No lesions detected in disease mask")
            features["lesion_count"] = 0
            features["total_disease_area"] = 0.0
            features["average_lesion_area"] = 0.0
            features["max_lesion_area"] = 0.0
            features["total_perimeter"] = 0.0
            features["average_perimeter"] = 0.0
            features["average_circularity"] = 0.0
            features["average_eccentricity"] = 0.0
            features["average_solidity"] = 0.0
            features["average_extent"] = 0.0
            return features

        # Use regionprops for detailed analysis
        regions = regionprops(labeled_array)

        # Basic count and area
        features["lesion_count"] = int(num_components)
        areas = [r.area for r in regions]
        features["total_disease_area"] = float(sum(areas))
        features["average_lesion_area"] = float(np.mean(areas))
        features["max_lesion_area"] = float(np.max(areas))

        # Perimeter stats
        perimeters = [r.perimeter for r in regions]
        features["total_perimeter"] = float(sum(perimeters))
        features["average_perimeter"] = float(np.mean(perimeters))

        # Circularity: 4π * area / perimeter²
        # (value of 1 = perfect circle)
        circularities = [
            4 * np.pi * (r.area / (r.perimeter ** 2 + 1e-8))
            for r in regions
        ]
        features["average_circularity"] = float(np.mean(circularities))

        # Eccentricity: (0=circle, 1=line)
        eccentricities = [r.eccentricity for r in regions]
        features["average_eccentricity"] = float(np.mean(eccentricities))

        # Solidity: area / convex_area
        solidities = [r.solidity for r in regions]
        features["average_solidity"] = float(np.mean(solidities))

        # Extent: area / bbox_area
        extents = [r.extent for r in regions]
        features["average_extent"] = float(np.mean(extents))

        return features


# =============================================================================
# Main Feature Extraction Function
# =============================================================================

def extract_features(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
    yellow_mask: Optional[np.ndarray] = None,
    brown_mask: Optional[np.ndarray] = None,
    gabor_frequencies: Tuple = (0.05, 0.1, 0.2),
    gabor_orientations: int = 4,
    gabor_sigma: float = 3.0,
    compute_leaf_colour_stats: bool = False,
) -> Dict[str, float]:
    """Extract complete feature set from a preprocessed image.

    Extracts texture features (Gabor filters), colour features (CIELAB),
    and morphology features (connected components) from the disease region.

    Parameters
    ----------
    image : np.ndarray
        BGR image (output of preprocessing pipeline).
    leaf_mask : np.ndarray
        Binary mask of leaf region (0 or 255).
    disease_mask : np.ndarray
        Binary mask of combined disease regions (0 or 255).
    yellow_mask : np.ndarray, optional
        Binary mask of yellow/chlorosis regions (0 or 255).
    brown_mask : np.ndarray, optional
        Binary mask of brown/necrosis regions (0 or 255).
    gabor_frequencies : tuple
        Spatial frequencies for Gabor filters (default: (0.05, 0.1, 0.2)).
    gabor_orientations : int
        Number of orientations for Gabor filters (default: 4).
    gabor_sigma : float
        Standard deviation of Gaussian envelope (default: 3.0).
    compute_leaf_colour_stats : bool
        If True, also compute LAB statistics for the leaf region.

    Returns
    -------
    dict
        Flat dictionary of feature names to numeric values.
        Feature groups:
          - gabor_mean_*: Gabor filter mean responses
          - gabor_std_*: Gabor filter std responses
          - gabor_energy_*: Gabor filter energy responses
          - disease/leaf l/a/b_mean/std: CIELAB colour stats (6-12 values)
          - *_ratio: Disease, yellow, brown pixel ratios (3 values)
          - lesion_count: Number of separate lesions (1 value)
          - *_area: Area statistics (4 values)
          - *_perimeter: Perimeter statistics (2 values)
          - average_circularity: Shape metric (1 value)
          - average_eccentricity: Shape metric (1 value)
          - average_solidity: Shape metric (1 value)
          - average_extent: Shape metric (1 value)

    Examples
    --------
    >>> from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
    >>> pipeline = PreprocessingPipeline()
    >>> result = pipeline.run(image)
    >>> features = extract_features(
    ...     image=result.shadow_removed,
    ...     leaf_mask=result.leaf_mask,
    ...     disease_mask=result.disease_mask,
    ...     yellow_mask=result.yellow_mask,
    ...     brown_mask=result.brown_mask,
    ... )
    >>> print(f"Extracted {len(features)} features")
    """
    features = {}

    # Texture features (Gabor filters)
    gabor_extractor = GaborTextureExtractor(
        frequencies=gabor_frequencies,
        orientations=gabor_orientations,
        sigma=gabor_sigma,
    )
    gabor_features = gabor_extractor.extract(image, disease_mask)
    features.update(gabor_features)

    # Colour features (CIELAB)
    colour_extractor = ColourFeatureExtractor(
        compute_leaf_stats=compute_leaf_colour_stats
    )
    colour_features = colour_extractor.extract(
        image,
        disease_mask=disease_mask,
        yellow_mask=yellow_mask,
        brown_mask=brown_mask,
        leaf_mask=leaf_mask,
    )
    features.update(colour_features)

    # Morphology features (connected components)
    morph_extractor = MorphologyFeatureExtractor()
    morph_features = morph_extractor.extract(disease_mask)
    features.update(morph_features)

    return features


# =============================================================================
# Inspection and Visualization Utilities
# =============================================================================

def extract_features_batch(
    image_list: list,
    leaf_mask_list: list,
    disease_mask_list: list,
    yellow_mask_list: Optional[list] = None,
    brown_mask_list: Optional[list] = None,
    image_ids: Optional[list] = None,
    **kwargs,
) -> "pd.DataFrame":
    """Extract features from multiple images and return as DataFrame.

    Parameters
    ----------
    image_list : list
        List of BGR images.
    leaf_mask_list : list
        List of leaf masks.
    disease_mask_list : list
        List of disease masks.
    yellow_mask_list : list, optional
        List of yellow masks.
    brown_mask_list : list, optional
        List of brown masks.
    image_ids : list, optional
        List of image identifiers/names for DataFrame index.
    **kwargs
        Additional keyword arguments passed to extract_features().

    Returns
    -------
    pd.DataFrame
        Features for all images with image_id as index.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas required for extract_features_batch(). "
            "Install with: pip install pandas"
        )

    n_images = len(image_list)
    if not all(
        len(lst) == n_images
        for lst in [leaf_mask_list, disease_mask_list]
    ):
        raise ValueError("All input lists must have the same length")

    if image_ids is None:
        image_ids = [f"image_{i:04d}" for i in range(n_images)]

    if len(image_ids) != n_images:
        raise ValueError("image_ids length must match image_list length")

    # Handle optional mask lists
    if yellow_mask_list is None:
        yellow_mask_list = [None] * n_images
    if brown_mask_list is None:
        brown_mask_list = [None] * n_images

    # Extract features for each image
    feature_dicts = []
    for i, (img, leaf_m, disease_m, yellow_m, brown_m, img_id) in enumerate(
        zip(
            image_list,
            leaf_mask_list,
            disease_mask_list,
            yellow_mask_list,
            brown_mask_list,
            image_ids,
        )
    ):
        try:
            features = extract_features(
                image=img,
                leaf_mask=leaf_m,
                disease_mask=disease_m,
                yellow_mask=yellow_m,
                brown_mask=brown_m,
                **kwargs,
            )
            features["image_id"] = img_id
            feature_dicts.append(features)
        except Exception as e:
            logger.error(f"Error extracting features for {img_id}: {e}")
            feature_dicts.append({"image_id": img_id, "error": str(e)})

    # Convert to DataFrame
    df = pd.DataFrame(feature_dicts)
    df.set_index("image_id", inplace=True)

    return df


def print_feature_summary(
    features: Dict[str, float],
    title: str = "Feature Extraction Summary",
) -> None:
    """Print a formatted summary of extracted features.

    Parameters
    ----------
    features : dict
        Feature dictionary from extract_features().
    title : str
        Title to display above the summary.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    # Group features by prefix
    groups = {}
    for name, value in features.items():
        prefix = name.rsplit("_", 1)[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((name, value))

    # Print each group
    for group_name in sorted(groups.keys()):
        print(f"\n{group_name.upper()}:")
        for name, value in sorted(groups[group_name]):
            if isinstance(value, float):
                print(f"  {name:40s} = {value:12.6f}")
            else:
                print(f"  {name:40s} = {value}")

    print("\n" + "=" * 70)
    print(f"Total features extracted: {len(features)}")
    print("=" * 70 + "\n")


# =============================================================================
# Integration with preprocessing pipeline
# =============================================================================

def extract_features_from_pipeline_result(
    pipeline_result,
    **kwargs,
) -> Dict[str, float]:
    """Extract features directly from a PipelineResult object.

    Convenience function that extracts features from the standard
    preprocessing pipeline output.

    Parameters
    ----------
    pipeline_result : PipelineResult
        Output from PreprocessingPipeline.run().
    **kwargs
        Additional keyword arguments passed to extract_features().

    Returns
    -------
    dict
        Feature dictionary.

    Examples
    --------
    >>> from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
    >>> pipeline = PreprocessingPipeline()
    >>> result = pipeline.run(image)
    >>> features = extract_features_from_pipeline_result(result)
    """
    return extract_features(
        image=pipeline_result.shadow_removed,
        leaf_mask=pipeline_result.leaf_mask,
        disease_mask=pipeline_result.disease_mask,
        yellow_mask=pipeline_result.yellow_mask,
        brown_mask=pipeline_result.brown_mask,
        **kwargs,
    )
