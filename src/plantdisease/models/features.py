"""
Feature extraction module for XGBoost classification.

Feature groups (96 total):
    - HSV 3D colour histogram:  4 H × 4 S × 4 V = 64 features
    - LBP (Local Binary Pattern): 24 points, radius 8, 26 bins (uniform)
    - GLCM (Gray-Level Co-occurrence Matrix): 8 levels, 6 texture features
      (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# HSV 3-D colour histogram  (64 features)
# -----------------------------------------------------------------------

class HSVHistogramExtractor:
    """Extract a 3-D HSV colour histogram.

    Hue:        4 bins  (0-45, 45-90, 90-135, 135-180)
    Saturation: 4 bins  (0-64, 64-128, 128-192, 192-256)
    Value:      4 bins  (0-64, 64-128, 128-192, 192-256)

    Total: 4 × 4 × 4 = 64 normalised features (sum = 1).
    """

    def __init__(
        self,
        h_bins: int = 4,
        s_bins: int = 4,
        v_bins: int = 4,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1, 2], mask,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-6)
        return hist

    @property
    def feature_dim(self) -> int:
        return self.h_bins * self.s_bins * self.v_bins

    @property
    def feature_names(self) -> List[str]:
        names = []
        for h in range(self.h_bins):
            for s in range(self.s_bins):
                for v in range(self.v_bins):
                    names.append(f"hsv_h{h}_s{s}_v{v}")
        return names


# -----------------------------------------------------------------------
# LBP histogram  (26 features)
# -----------------------------------------------------------------------

class LBPExtractor:
    """Extract Local Binary Pattern texture features.

    24 sampling points, radius 8, uniform patterns.
    Produces a 26-bin histogram (25 uniform + 1 non-uniform).
    """

    def __init__(
        self,
        radius: int = 8,
        n_points: int = 24,
        method: str = "uniform",
        n_bins: int = 26,
    ):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.n_bins = n_bins

        self._use_skimage = False
        try:
            from skimage.feature import local_binary_pattern
            self._local_binary_pattern = local_binary_pattern
            self._use_skimage = True
        except ImportError:
            pass

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self._use_skimage:
            lbp = self._local_binary_pattern(
                gray, self.n_points, self.radius, method=self.method,
            )
        else:
            lbp = self._compute_lbp(gray)

        if mask is not None:
            lbp_vals = lbp[mask > 0]
        else:
            lbp_vals = lbp.flatten()

        hist, _ = np.histogram(
            lbp_vals, bins=self.n_bins, range=(0, self.n_bins),
        )
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        return hist

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Pure-Python LBP fallback (slower)."""
        rows, cols = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for i in range(self.radius, rows - self.radius):
            for j in range(self.radius, cols - self.radius):
                centre = gray[i, j]
                code = 0
                for k in range(8):
                    angle = 2 * np.pi * k / 8
                    ni = int(round(i + self.radius * np.sin(angle)))
                    nj = int(round(j + self.radius * np.cos(angle)))
                    if gray[ni, nj] >= centre:
                        code |= 1 << k
                lbp[i, j] = code
        return lbp

    @property
    def feature_dim(self) -> int:
        return self.n_bins

    @property
    def feature_names(self) -> List[str]:
        names = [f"lbp_uniform_{i}" for i in range(self.n_bins - 1)]
        names.append("lbp_non_uniform")
        return names


# -----------------------------------------------------------------------
# GLCM texture features  (6 features)
# -----------------------------------------------------------------------

class GLCMExtractor:
    """Extract Gray-Level Co-occurrence Matrix (GLCM) texture features.
    The grayscale image is quantised to 8 intensity levels.  A single
    GLCM is computed (distance=1, angle=0) and six statistical measures
    are extracted: contrast, dissimilarity, homogeneity, energy,
    correlation, and ASM.
    """

    PROPERTIES = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    def __init__(self, levels: int = 8):
        self.levels = levels

        self._use_skimage = False
        try:
            from skimage.feature import graycomatrix, graycoprops
            self._graycomatrix = graycomatrix
            self._graycoprops = graycoprops
            self._use_skimage = True
        except ImportError:
            pass

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Quantise to `levels` intensity levels
        gray = (gray.astype(np.float32) / 256 * self.levels).astype(np.uint8)

        if self._use_skimage:
            return self._extract_skimage(gray)
        return self._extract_pure(gray)

    # -- scikit-image path ------------------------------------------------

    def _extract_skimage(self, gray: np.ndarray) -> np.ndarray:
        glcm = self._graycomatrix(
            gray,
            distances=[1],
            angles=[0],
            levels=self.levels,
            symmetric=True,
            normed=True,
        )
        feats = []
        for prop in ("contrast", "dissimilarity", "homogeneity",
                     "energy", "correlation"):
            feats.append(float(self._graycoprops(glcm, prop)[0, 0]))
        # ASM = energy²
        feats.append(feats[3] ** 2)
        return np.array(feats, dtype=np.float64)

    # -- pure-Python fallback ---------------------------------------------

    def _extract_pure(self, gray: np.ndarray) -> np.ndarray:
        glcm = self._compute_glcm(gray)
        props = self._compute_properties(glcm)
        return np.array([props[p] for p in self.PROPERTIES], dtype=np.float64)

    def _compute_glcm(self, gray: np.ndarray) -> np.ndarray:
        rows, cols = gray.shape
        glcm = np.zeros((self.levels, self.levels), dtype=np.float32)
        for i in range(rows):
            for j in range(cols - 1):
                glcm[gray[i, j], gray[i, j + 1]] += 1
        glcm = glcm + glcm.T
        glcm = glcm / (glcm.sum() + 1e-6)
        return glcm

    @staticmethod
    def _compute_properties(glcm: np.ndarray) -> Dict[str, float]:
        n = glcm.shape[0]
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mu_i = float(np.sum(I * glcm))
        mu_j = float(np.sum(J * glcm))
        sigma_i = float(np.sqrt(np.sum((I - mu_i) ** 2 * glcm)))
        sigma_j = float(np.sqrt(np.sum((J - mu_j) ** 2 * glcm)))

        contrast = float(np.sum((I - J) ** 2 * glcm))
        dissimilarity = float(np.sum(np.abs(I - J) * glcm))
        homogeneity = float(np.sum(glcm / (1 + (I - J) ** 2)))
        energy = float(np.sqrt(np.sum(glcm ** 2)))
        asm = float(np.sum(glcm ** 2))
        if sigma_i > 0 and sigma_j > 0:
            correlation = float(
                np.sum((I - mu_i) * (J - mu_j) * glcm) / (sigma_i * sigma_j)
            )
        else:
            correlation = 0.0

        return {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "ASM": asm,
        }

    @property
    def feature_dim(self) -> int:
        return 6

    @property
    def feature_names(self) -> List[str]:
        return [f"glcm_{p.lower()}" for p in self.PROPERTIES]


# -----------------------------------------------------------------------
# Combined 96-feature extractor
# -----------------------------------------------------------------------


class FeatureExtractor:
    """

    Extracts the segmented leaf image into HSV space (64 colour
    features) and grayscale (26 LBP + 6 GLCM = 32 texture features).
    Total: 64 + 26 + 6 = 96 features.
    """

    def __init__(self):
        self.hsv_extractor = HSVHistogramExtractor()   # 64
        self.lbp_extractor = LBPExtractor()            # 26
        self.glcm_extractor = GLCMExtractor()          # 6

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 1. HSV colour histogram (from the colour image)
        hsv_features = self.hsv_extractor.extract(image, mask)

        # 2. LBP texture (internally converts to grayscale)
        lbp_features = self.lbp_extractor.extract(image, mask)

        # 3. GLCM texture (internally converts to grayscale, quantises to 8 levels)
        glcm_features = self.glcm_extractor.extract(image, mask)

        return np.concatenate([hsv_features, lbp_features, glcm_features])

    @property
    def feature_dim(self) -> int:
        return (
            self.hsv_extractor.feature_dim
            + self.lbp_extractor.feature_dim
            + self.glcm_extractor.feature_dim
        )

    @property
    def feature_names(self) -> List[str]:
        return (
            self.hsv_extractor.feature_names
            + self.lbp_extractor.feature_names
            + self.glcm_extractor.feature_names
        )


def extract_features_from_directory(
    input_dir: Union[str, Path],
    extractor: Optional[FeatureExtractor] = None,
    mask_dir: Optional[Union[str, Path]] = None,
    extensions: List[str] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract features from all images in a directory.
    
    Args:
        input_dir: Directory containing images
        extractor: FeatureExtractor instance
        mask_dir: Optional directory containing masks
        extensions: Valid file extensions
    
    Returns:
        Tuple of (feature_matrix, image_names, label_list)
    """
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    if extractor is None:
        extractor = FeatureExtractor()
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.rglob(f'*{ext}'))
        image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    features_list = []
    image_names = []
    labels = []
    
    for image_path in tqdm(image_files, desc="Extracting features"):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            continue
        
        # Load mask if available
        mask = None
        if mask_dir:
            mask_path = Path(mask_dir) / f"{image_path.stem}_leaf_mask.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Extract features
        features = extractor.extract(image, mask)
        features_list.append(features)
        
        image_names.append(image_path.name)
        
        # Infer label from parent directory
        labels.append(image_path.parent.name)
    
    # Stack features
    feature_matrix = np.vstack(features_list)
    
    logger.info(f"Extracted features from {len(image_names)} images, shape: {feature_matrix.shape}")
    
    return feature_matrix, image_names, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="96-feature extraction",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input directory with class subfolders")
    parser.add_argument("--output", "-o", required=True,
                        help="Output NPZ file")
    parser.add_argument("--mask-dir", help="Directory containing masks")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    extractor = FeatureExtractor()

    features, names, labels = extract_features_from_directory(
        args.input, extractor, args.mask_dir,
    )

    np.savez(
        args.output,
        features=features,
        image_names=names,
        labels=labels,
        feature_names=extractor.feature_names,
    )

    print(f"Saved features to {args.output}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature groups: 64 HSV + 26 LBP + 6 GLCM = {extractor.feature_dim}")
