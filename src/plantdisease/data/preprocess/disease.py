"""
Diseased region segmentation module using HSV color analysis.

This module extracts diseased regions from segmented leaf images using
HSV color thresholds to detect:
- Yellow regions (chlorosis)
- Brown regions (necrosis/rot)
- Adjacent green regions near lesions

Also computes disease severity metrics.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HSVRange:
    """HSV color range for thresholding."""
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int
    
    def to_lower(self) -> np.ndarray:
        """Get lower bound array."""
        return np.array([self.h_min, self.s_min, self.v_min])
    
    def to_upper(self) -> np.ndarray:
        """Get upper bound array."""
        return np.array([self.h_max, self.s_max, self.v_max])


@dataclass
class SeverityMetrics:
    """Disease severity metrics."""
    total_leaf_pixels: int
    diseased_pixels: int
    yellow_pixels: int
    brown_pixels: int
    severity_percent: float
    yellow_severity_percent: float
    brown_severity_percent: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# Default HSV ranges for common disease symptoms
# Adjusted to avoid shadow false positives
DEFAULT_YELLOW_RANGE = HSVRange(
    h_min=15, h_max=35,
    s_min=70, s_max=255,  # Higher saturation to exclude shadows
    v_min=80, v_max=255   # Higher value to exclude dark shadows
)

DEFAULT_BROWN_RANGE = HSVRange(
    h_min=0, h_max=25,
    s_min=60, s_max=200,  # Require decent saturation (shadows are desaturated)
    v_min=50, v_max=180   # Raised minimum to exclude shadows
)

DEFAULT_GREEN_RANGE = HSVRange(
    h_min=35, h_max=85,
    s_min=40, s_max=255,
    v_min=40, v_max=255
)


class DiseaseSegmenter:
    """
    Segment diseased regions from leaf images using HSV color analysis.
    
    Detects yellow (chlorosis) and brown (necrosis) regions, optionally
    including adjacent green regions that may be transitioning.
    """
    
    def __init__(
        self,
        yellow_range: Optional[HSVRange] = None,
        brown_range: Optional[HSVRange] = None,
        green_range: Optional[HSVRange] = None,
        include_adjacent_green: bool = True,
        adjacent_dilation_kernel: int = 5,
        morph_kernel_size: int = 3,
        min_contour_area: int = 50
    ):
        """
        Initialize disease segmenter.
        
        Args:
            yellow_range: HSV range for yellow/chlorosis detection
            brown_range: HSV range for brown/necrosis detection
            green_range: HSV range for healthy green tissue
            include_adjacent_green: Include green regions adjacent to lesions
            adjacent_dilation_kernel: Kernel size for dilating disease mask
            morph_kernel_size: Kernel size for morphological operations
            min_contour_area: Minimum area for disease contours
        """
        self.yellow_range = yellow_range or DEFAULT_YELLOW_RANGE
        self.brown_range = brown_range or DEFAULT_BROWN_RANGE
        self.green_range = green_range or DEFAULT_GREEN_RANGE
        self.include_adjacent_green = include_adjacent_green
        self.adjacent_dilation_kernel = adjacent_dilation_kernel
        self.morph_kernel_size = morph_kernel_size
        self.min_contour_area = min_contour_area
        
        # Morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )
    
    def detect_yellow_regions(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Detect yellow/chlorotic regions.
        
        Args:
            hsv_image: Image in HSV format
            
        Returns:
            Binary mask of yellow regions
        """
        mask = cv2.inRange(
            hsv_image,
            self.yellow_range.to_lower(),
            self.yellow_range.to_upper()
        )
        return mask
    
    def detect_brown_regions(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Detect brown/necrotic regions.
        
        Brown can span hue=0 (red) which wraps around, so we may need
        to handle two ranges. Excludes very dark/low-saturation regions
        that are likely shadows.
        
        Args:
            hsv_image: Image in HSV format
            
        Returns:
            Binary mask of brown regions
        """
        # Standard brown range (requires decent saturation to exclude shadows)
        mask = cv2.inRange(
            hsv_image,
            self.brown_range.to_lower(),
            self.brown_range.to_upper()
        )
        
        # Also check reddish-brown with high hue (wraps around 180)
        # But require good saturation to distinguish from shadows
        reddish_brown_lower = np.array([165, 60, 50])
        reddish_brown_upper = np.array([180, 200, 180])
        reddish_mask = cv2.inRange(hsv_image, reddish_brown_lower, reddish_brown_upper)
        
        combined = cv2.bitwise_or(mask, reddish_mask)
        
        # Exclude very low saturation regions (likely shadows, not disease)
        s_channel = hsv_image[:, :, 1]
        v_channel = hsv_image[:, :, 2]
        shadow_like = ((s_channel < 50) | (v_channel < 40)).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(shadow_like))
        
        return combined
    
    def detect_green_regions(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Detect healthy green regions.
        
        Args:
            hsv_image: Image in HSV format
            
        Returns:
            Binary mask of green regions
        """
        mask = cv2.inRange(
            hsv_image,
            self.green_range.to_lower(),
            self.green_range.to_upper()
        )
        return mask
    
    def get_adjacent_green(
        self,
        disease_mask: np.ndarray,
        green_mask: np.ndarray
    ) -> np.ndarray:
        """
        Get green regions adjacent to diseased areas.
        
        This captures transition zones where disease is spreading.
        
        Args:
            disease_mask: Combined disease mask
            green_mask: Healthy green mask
            
        Returns:
            Mask of green regions adjacent to disease
        """
        # Dilate disease mask
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.adjacent_dilation_kernel, self.adjacent_dilation_kernel)
        )
        dilated = cv2.dilate(disease_mask, dilation_kernel, iterations=2)
        
        # Find overlap with green mask (green adjacent to disease)
        adjacent = cv2.bitwise_and(dilated, green_mask)
        
        # Remove the original disease regions
        adjacent = cv2.bitwise_and(adjacent, cv2.bitwise_not(disease_mask))
        
        return adjacent
    
    def cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological cleanup to remove noise and fill holes.
        
        Args:
            mask: Input binary mask
            
        Returns:
            Cleaned mask
        """
        # Remove small noise with opening
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Fill small holes with closing
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Remove small contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.drawContours(cleaned, [contour], -1, 255, -1)
        
        return cleaned
    
    def segment_disease(
        self,
        image: np.ndarray,
        leaf_mask: Optional[np.ndarray] = None,
        shadow_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SeverityMetrics]:
        """
        Segment diseased regions from leaf image.
        
        Args:
            image: Input image in BGR format (should be segmented leaf)
            leaf_mask: Optional mask indicating valid leaf pixels
            shadow_mask: Optional mask of shadow regions to exclude
            
        Returns:
            Tuple of (combined_disease_mask, yellow_mask, brown_mask, severity_metrics)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect individual symptom types
        yellow_mask = self.detect_yellow_regions(hsv)
        brown_mask = self.detect_brown_regions(hsv)
        
        # Combine disease masks
        disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)
        
        # Exclude shadow regions if provided
        if shadow_mask is not None:
            inverse_shadow = cv2.bitwise_not(shadow_mask)
            disease_mask = cv2.bitwise_and(disease_mask, inverse_shadow)
            yellow_mask = cv2.bitwise_and(yellow_mask, inverse_shadow)
            brown_mask = cv2.bitwise_and(brown_mask, inverse_shadow)
        
        # Optionally include adjacent green regions
        if self.include_adjacent_green:
            green_mask = self.detect_green_regions(hsv)
            adjacent_green = self.get_adjacent_green(disease_mask, green_mask)
            # Also exclude shadows from adjacent green
            if shadow_mask is not None:
                adjacent_green = cv2.bitwise_and(adjacent_green, inverse_shadow)
            disease_mask = cv2.bitwise_or(disease_mask, adjacent_green)
        
        # Apply leaf mask if provided
        if leaf_mask is not None:
            disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)
            yellow_mask = cv2.bitwise_and(yellow_mask, leaf_mask)
            brown_mask = cv2.bitwise_and(brown_mask, leaf_mask)
        
        # Cleanup masks
        disease_mask = self.cleanup_mask(disease_mask)
        yellow_mask = self.cleanup_mask(yellow_mask)
        brown_mask = self.cleanup_mask(brown_mask)
        
        # Compute severity metrics
        if leaf_mask is not None:
            total_leaf_pixels = int(np.sum(leaf_mask > 0))
        else:
            # Estimate leaf pixels from non-black pixels
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            total_leaf_pixels = int(np.sum(gray > 10))
        
        diseased_pixels = int(np.sum(disease_mask > 0))
        yellow_pixels = int(np.sum(yellow_mask > 0))
        brown_pixels = int(np.sum(brown_mask > 0))
        
        if total_leaf_pixels > 0:
            severity_percent = (diseased_pixels / total_leaf_pixels) * 100
            yellow_severity = (yellow_pixels / total_leaf_pixels) * 100
            brown_severity = (brown_pixels / total_leaf_pixels) * 100
        else:
            severity_percent = 0.0
            yellow_severity = 0.0
            brown_severity = 0.0
        
        metrics = SeverityMetrics(
            total_leaf_pixels=total_leaf_pixels,
            diseased_pixels=diseased_pixels,
            yellow_pixels=yellow_pixels,
            brown_pixels=brown_pixels,
            severity_percent=round(severity_percent, 2),
            yellow_severity_percent=round(yellow_severity, 2),
            brown_severity_percent=round(brown_severity, 2)
        )
        
        return disease_mask, yellow_mask, brown_mask, metrics
    
    def create_disease_overlay(
        self,
        image: np.ndarray,
        disease_mask: np.ndarray,
        yellow_mask: np.ndarray,
        brown_mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization overlay showing disease regions.
        
        Color coding:
        - Yellow regions: Yellow overlay
        - Brown regions: Red overlay
        - Other disease: Orange overlay
        
        Args:
            image: Original image
            disease_mask: Combined disease mask
            yellow_mask: Yellow/chlorosis mask
            brown_mask: Brown/necrosis mask
            alpha: Overlay transparency
            
        Returns:
            Overlay visualization image
        """
        overlay = image.copy()
        
        # Yellow regions in yellow
        yellow_color = np.array([0, 255, 255])  # BGR
        mask = yellow_mask > 0
        overlay[mask] = (
            (1 - alpha) * overlay[mask] + alpha * yellow_color
        ).astype(np.uint8)
        
        # Brown regions in red
        brown_color = np.array([0, 0, 255])  # BGR
        mask = brown_mask > 0
        overlay[mask] = (
            (1 - alpha) * overlay[mask] + alpha * brown_color
        ).astype(np.uint8)
        
        return overlay


def segment_disease_from_image(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    leaf_mask_path: Optional[Union[str, Path]] = None,
    segmenter: Optional[DiseaseSegmenter] = None
) -> SeverityMetrics:
    """
    Segment diseased regions from an image file and save outputs.
    
    Args:
        image_path: Path to input image (segmented leaf)
        output_dir: Directory to save outputs
        leaf_mask_path: Path to leaf mask (optional)
        segmenter: DiseaseSegmenter instance
        
    Returns:
        SeverityMetrics
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if segmenter is None:
        segmenter = DiseaseSegmenter()
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Load leaf mask if provided
    leaf_mask = None
    if leaf_mask_path:
        leaf_mask = cv2.imread(str(leaf_mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Segment disease
    disease_mask, yellow_mask, brown_mask, metrics = segmenter.segment_disease(
        image, leaf_mask
    )
    
    # Define output paths
    stem = image_path.stem
    disease_mask_path = output_dir / f"{stem}_diseased_mask.png"
    overlay_path = output_dir / f"{stem}_diseased_overlay.png"
    metrics_path = output_dir / f"{stem}_severity_metrics.json"
    
    # Save disease mask
    cv2.imwrite(str(disease_mask_path), disease_mask)
    
    # Save overlay
    overlay = segmenter.create_disease_overlay(
        image, disease_mask, yellow_mask, brown_mask
    )
    cv2.imwrite(str(overlay_path), overlay)
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    logger.info(
        f"Disease segmentation for {image_path.name}: "
        f"severity={metrics.severity_percent:.1f}%"
    )
    
    return metrics


def batch_segment_disease(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    segmenter: Optional[DiseaseSegmenter] = None,
    extensions: List[str] = None
) -> Dict:
    """
    Batch segment disease from all images in a directory.
    
    Args:
        input_dir: Directory containing segmented leaf images
        output_dir: Directory to save outputs
        mask_dir: Directory containing leaf masks (optional)
        segmenter: DiseaseSegmenter instance
        extensions: List of valid extensions
        
    Returns:
        Dictionary with statistics
    """
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    if segmenter is None:
        segmenter = DiseaseSegmenter()
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    # Statistics
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
        'severity_distribution': {
            'healthy': 0,      # < 5%
            'mild': 0,         # 5-15%
            'moderate': 0,     # 15-30%
            'severe': 0        # > 30%
        },
        'average_severity': 0.0,
        'metrics': []
    }
    
    total_severity = 0.0
    
    for image_path in tqdm(image_files, desc="Segmenting disease"):
        try:
            # Find corresponding mask if mask_dir provided
            leaf_mask_path = None
            if mask_dir:
                mask_name = image_path.stem + "_leaf_mask.png"
                potential_mask = Path(mask_dir) / mask_name
                if potential_mask.exists():
                    leaf_mask_path = potential_mask
            
            metrics = segment_disease_from_image(
                image_path, output_dir, leaf_mask_path, segmenter
            )
            
            stats['processed'] += 1
            stats['metrics'].append({
                'image': image_path.name,
                **metrics.to_dict()
            })
            
            total_severity += metrics.severity_percent
            
            # Categorize severity
            if metrics.severity_percent < 5:
                stats['severity_distribution']['healthy'] += 1
            elif metrics.severity_percent < 15:
                stats['severity_distribution']['mild'] += 1
            elif metrics.severity_percent < 30:
                stats['severity_distribution']['moderate'] += 1
            else:
                stats['severity_distribution']['severe'] += 1
                
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            stats['failed'] += 1
    
    # Compute average severity
    if stats['processed'] > 0:
        stats['average_severity'] = round(total_severity / stats['processed'], 2)
    
    # Save statistics
    stats_path = output_dir / 'disease_segmentation_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Batch disease segmentation complete: {stats['processed']}/{stats['total']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Disease region segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--mask-dir", help="Directory containing leaf masks")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        metrics = segment_disease_from_image(input_path, args.output)
        print(f"Severity: {metrics.severity_percent:.1f}%")
    else:
        stats = batch_segment_disease(input_path, args.output, args.mask_dir)
        print(f"Processed: {stats['processed']}, Average severity: {stats['average_severity']:.1f}%")
