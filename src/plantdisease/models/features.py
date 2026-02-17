"""
Feature extraction module for traditional ML baseline (XGBoost).

Extracts features from segmented leaf images:
- HSV color histogram features
- LBP (Local Binary Pattern) texture features
- GLCM (Gray Level Co-occurrence Matrix) texture features

Note: These features are based on established plant disease
detection literature workflows.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class HSVHistogramExtractor:
    """Extract HSV color histogram features."""
    
    def __init__(
        self,
        h_bins: int = 32,
        s_bins: int = 32,
        v_bins: int = 32,
        normalize: bool = True
    ):
        """
        Initialize HSV histogram extractor.
        
        Args:
            h_bins: Number of bins for Hue channel
            s_bins: Number of bins for Saturation channel
            v_bins: Number of bins for Value channel
            normalize: Whether to normalize histograms
        """
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.normalize = normalize
    
    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract HSV histogram features.
        
        Args:
            image: Input image in BGR format
            mask: Optional mask (non-zero pixels are foreground)
        
        Returns:
            Feature vector (h_bins + s_bins + v_bins,)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        h_hist = cv2.calcHist(
            [hsv], [0], mask, [self.h_bins], [0, 180]
        ).flatten()
        
        s_hist = cv2.calcHist(
            [hsv], [1], mask, [self.s_bins], [0, 256]
        ).flatten()
        
        v_hist = cv2.calcHist(
            [hsv], [2], mask, [self.v_bins], [0, 256]
        ).flatten()
        
        # Concatenate
        features = np.concatenate([h_hist, s_hist, v_hist])
        
        # Normalize
        if self.normalize:
            features = features / (np.sum(features) + 1e-6)
        
        return features
    
    @property
    def feature_dim(self) -> int:
        """Return feature vector dimension."""
        return self.h_bins + self.s_bins + self.v_bins
    
    @property
    def feature_names(self) -> List[str]:
        """Return feature names."""
        names = []
        for i in range(self.h_bins):
            names.append(f'hsv_h_bin_{i}')
        for i in range(self.s_bins):
            names.append(f'hsv_s_bin_{i}')
        for i in range(self.v_bins):
            names.append(f'hsv_v_bin_{i}')
        return names


class LBPExtractor:
    """Extract Local Binary Pattern texture features.
    
    Uses scikit-image for fast computation when available.
    """
    
    def __init__(
        self,
        radius: int = 3,
        n_points: int = 24,
        method: str = 'uniform',
        n_bins: int = 26
    ):
        """
        Initialize LBP extractor.
        
        Args:
            radius: Radius of circular LBP
            n_points: Number of circular neighbors
            method: LBP method ('uniform', 'default')
            n_bins: Number of histogram bins
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.n_bins = n_bins
        
        # Try to use skimage for faster computation
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
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract LBP histogram features.
        
        Args:
            image: Input image (BGR or grayscale)
            mask: Optional mask (non-zero pixels are foreground)
        
        Returns:
            Feature vector (n_bins,)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute LBP
        if self._use_skimage:
            lbp = self._local_binary_pattern(gray, self.n_points, self.radius, method=self.method)
        else:
            lbp = self._compute_lbp(gray)
        
        # Compute histogram
        if mask is not None:
            lbp_masked = lbp[mask > 0]
            hist, _ = np.histogram(lbp_masked, bins=self.n_bins, range=(0, self.n_bins))
        else:
            hist, _ = np.histogram(lbp.flatten(), bins=self.n_bins, range=(0, self.n_bins))
        
        # Normalize
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-6)
        
        return hist
    
    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Compute LBP image (pure Python fallback)."""
        rows, cols = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        
        for i in range(self.radius, rows - self.radius):
            for j in range(self.radius, cols - self.radius):
                center = gray[i, j]
                code = 0
                for k in range(8):
                    angle = 2 * np.pi * k / 8
                    ni = int(round(i + self.radius * np.sin(angle)))
                    nj = int(round(j + self.radius * np.cos(angle)))
                    if gray[ni, nj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        return lbp
    
    @property
    def feature_dim(self) -> int:
        """Return feature vector dimension."""
        return self.n_bins
    
    @property
    def feature_names(self) -> List[str]:
        """Return feature names."""
        return [f'lbp_bin_{i}' for i in range(self.n_bins)]


class GLCMExtractor:
    """Extract GLCM (Gray Level Co-occurrence Matrix) texture features.
    
    Uses scikit-image for fast computation when available.
    """
    
    def __init__(
        self,
        distances: List[int] = None,
        angles: List[float] = None,
        levels: int = 32,  # Reduced for speed
        symmetric: bool = True,
        normed: bool = True
    ):
        """
        Initialize GLCM extractor.
        
        Args:
            distances: Pixel pair distances (default: [1, 2, 3])
            angles: Angles in radians (default: [0, π/4, π/2, 3π/4])
            levels: Number of gray levels (will quantize image)
            symmetric: If True, GLCM is symmetric
            normed: If True, normalize GLCM
        """
        self.distances = distances or [1, 2, 3]
        self.angles = angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        
        # Try to use skimage for faster computation
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
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract GLCM texture features.
        
        Args:
            image: Input image (BGR or grayscale)
            mask: Optional mask (not fully supported yet)
        
        Returns:
            Feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Quantize to reduce levels
        gray = (gray / 256 * self.levels).astype(np.uint8)
        
        if self._use_skimage:
            return self._extract_skimage(gray)
        else:
            return self._extract_pure_python(gray)
    
    def _extract_skimage(self, gray: np.ndarray) -> np.ndarray:
        """Extract GLCM features using scikit-image (fast)."""
        # Compute GLCM
        glcm = self._graycomatrix(
            gray, 
            distances=self.distances, 
            angles=self.angles,
            levels=self.levels,
            symmetric=self.symmetric,
            normed=self.normed
        )
        
        # Extract properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        all_features = []
        for prop in properties:
            values = self._graycoprops(glcm, prop)
            all_features.extend(values.flatten())
        
        # Add entropy manually (not in skimage)
        for d_idx in range(len(self.distances)):
            for a_idx in range(len(self.angles)):
                glcm_slice = glcm[:, :, d_idx, a_idx]
                glcm_nz = glcm_slice[glcm_slice > 0]
                entropy = -np.sum(glcm_nz * np.log2(glcm_nz + 1e-10))
                all_features.append(entropy)
        
        return np.array(all_features)
    
    def _extract_pure_python(self, gray: np.ndarray) -> np.ndarray:
        """Extract GLCM features using pure Python (slower fallback)."""
        all_features = []
        
        for distance in self.distances:
            for angle in self.angles:
                glcm = self._compute_glcm(gray, distance, angle)
                props = self._compute_properties(glcm)
                all_features.extend(props.values())
        
        return np.array(all_features)
    
    def _compute_glcm(self, gray: np.ndarray, distance: int, angle: float) -> np.ndarray:
        """Compute GLCM for single distance and angle (pure Python)."""
        rows, cols = gray.shape
        dx = int(round(distance * np.cos(angle)))
        dy = int(round(distance * np.sin(angle)))
        
        glcm = np.zeros((self.levels, self.levels), dtype=np.float32)
        
        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                i2 = i + dy
                j2 = j + dx
                if 0 <= i2 < rows and 0 <= j2 < cols:
                    glcm[gray[i, j], gray[i2, j2]] += 1
        
        if self.symmetric:
            glcm = glcm + glcm.T
        if self.normed:
            glcm = glcm / (np.sum(glcm) + 1e-6)
        
        return glcm
    
    def _compute_properties(self, glcm: np.ndarray) -> Dict[str, float]:
        """Compute GLCM texture properties."""
        i = np.arange(self.levels)
        j = np.arange(self.levels)
        I, J = np.meshgrid(i, j, indexing='ij')
        
        mu_i = np.sum(I * glcm)
        mu_j = np.sum(J * glcm)
        sigma_i = np.sqrt(np.sum((I - mu_i)**2 * glcm))
        sigma_j = np.sqrt(np.sum((J - mu_j)**2 * glcm))
        
        contrast = np.sum((I - J)**2 * glcm)
        dissimilarity = np.sum(np.abs(I - J) * glcm)
        homogeneity = np.sum(glcm / (1 + (I - J)**2))
        energy = np.sum(glcm**2)
        
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((I - mu_i) * (J - mu_j) * glcm) / (sigma_i * sigma_j)
        else:
            correlation = 0
        
        glcm_nz = glcm[glcm > 0]
        entropy = -np.sum(glcm_nz * np.log2(glcm_nz + 1e-10))
        
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation,
            'entropy': entropy
        }
    
    @property
    def feature_dim(self) -> int:
        """Return feature vector dimension."""
        n_combinations = len(self.distances) * len(self.angles)
        n_properties = 6  # contrast, dissimilarity, homogeneity, energy, correlation, entropy
        return n_combinations * n_properties
    
    @property
    def feature_names(self) -> List[str]:
        """Return feature names."""
        names = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                      'energy', 'correlation', 'entropy']
        
        for d_idx, distance in enumerate(self.distances):
            for a_idx, angle in enumerate(self.angles):
                for prop in properties:
                    names.append(f'glcm_d{distance}_a{a_idx}_{prop}')
        
        return names


class FeatureExtractor:
    """
    Combined feature extractor for plant disease classification.
    
    Extracts and concatenates features from multiple extractors.
    Supports optional grayscale preprocessing.
    """
    
    def __init__(
        self,
        hsv_bins: int = 32,
        lbp_radius: int = 3,
        lbp_n_points: int = 24,
        lbp_n_bins: int = 26,
        glcm_distances: List[int] = None,
        glcm_angles: List[float] = None,
        glcm_levels: int = 32,
        use_grayscale: bool = False,
        grayscale_method: str = 'luminosity'
    ):
        """
        Initialize combined feature extractor.
        
        Args:
            hsv_bins: Bins per HSV channel
            lbp_radius: LBP radius
            lbp_n_points: LBP neighbor points
            lbp_n_bins: LBP histogram bins
            glcm_distances: GLCM distances
            glcm_angles: GLCM angles
            glcm_levels: GLCM quantization levels
            use_grayscale: If True, convert images to grayscale before extraction.
                          Note: HSV features will be extracted from the 3-channel
                          grayscale version which will have uniform H/S values.
            grayscale_method: Method for grayscale conversion ('luminosity', 
                            'average', 'lightness', 'opencv')
        """
        self.use_grayscale = use_grayscale
        self.grayscale_method = grayscale_method
        
        self.hsv_extractor = HSVHistogramExtractor(
            h_bins=hsv_bins, s_bins=hsv_bins, v_bins=hsv_bins
        )
        
        self.lbp_extractor = LBPExtractor(
            radius=lbp_radius, n_points=lbp_n_points, n_bins=lbp_n_bins
        )
        
        self.glcm_extractor = GLCMExtractor(
            distances=glcm_distances or [1, 2, 3],
            angles=glcm_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=glcm_levels
        )
    
    def _preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if use_grayscale is enabled.
        
        Returns a 3-channel grayscale image for compatibility with
        color-based feature extractors.
        """
        if not self.use_grayscale:
            return image
        
        # Import here to avoid circular imports
        from src.plantdisease.data.preprocess.grayscale import to_grayscale
        
        # Convert to grayscale but keep 3 channels for HSV compatibility
        gray_3ch = to_grayscale(
            image, 
            method=self.grayscale_method,
            keep_channels=True
        )
        
        if gray_3ch is None:
            logger.warning("Grayscale conversion failed, using original image")
            return image
            
        return gray_3ch
    
    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract all features from image.
        
        Args:
            image: Input image in BGR format
            mask: Optional foreground mask
        
        Returns:
            Combined feature vector
        """
        # Apply grayscale preprocessing if enabled
        processed_image = self._preprocess_grayscale(image)
        
        # HSV histogram (on color/grayscale image)
        hsv_features = self.hsv_extractor.extract(processed_image, mask)
        
        # LBP (on grayscale - extractor handles conversion)
        lbp_features = self.lbp_extractor.extract(processed_image, mask)
        
        # GLCM (on quantized grayscale - extractor handles conversion)
        glcm_features = self.glcm_extractor.extract(processed_image, mask)
        
        # Concatenate all features
        features = np.concatenate([hsv_features, lbp_features, glcm_features])
        
        return features
    
    @property
    def feature_dim(self) -> int:
        """Return total feature dimension."""
        return (
            self.hsv_extractor.feature_dim +
            self.lbp_extractor.feature_dim +
            self.glcm_extractor.feature_dim
        )
    
    @property
    def feature_names(self) -> List[str]:
        """Return all feature names."""
        return (
            self.hsv_extractor.feature_names +
            self.lbp_extractor.feature_names +
            self.glcm_extractor.feature_names
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
    
    parser = argparse.ArgumentParser(description="Feature extraction for XGBoost baseline")
    parser.add_argument("--input", "-i", required=True, help="Input directory with class subfolders")
    parser.add_argument("--output", "-o", required=True, help="Output NPZ file")
    parser.add_argument("--mask-dir", help="Directory containing masks")
    parser.add_argument("--grayscale", action="store_true", 
                       help="Convert images to grayscale before feature extraction")
    parser.add_argument("--grayscale-method", default="luminosity",
                       choices=["luminosity", "average", "lightness", "opencv"],
                       help="Grayscale conversion method (default: luminosity)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    extractor = FeatureExtractor(
        use_grayscale=args.grayscale,
        grayscale_method=args.grayscale_method
    )
    
    if args.grayscale:
        logger.info(f"Grayscale preprocessing enabled (method: {args.grayscale_method})")
    
    features, names, labels = extract_features_from_directory(
        args.input, extractor, args.mask_dir
    )
    
    # Save to NPZ
    np.savez(
        args.output,
        features=features,
        image_names=names,
        labels=labels,
        feature_names=extractor.feature_names
    )
    
    print(f"Saved features to {args.output}")
    print(f"Feature shape: {features.shape}")
    print(f"Grayscale preprocessing: {'enabled' if args.grayscale else 'disabled'}")
