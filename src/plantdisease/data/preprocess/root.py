"""
Root image preprocessing module.

Root images are more challenging than leaf images due to:
- Variable soil/background conditions
- Complex root structures
- Inconsistent lighting

This module provides baseline root preprocessing with:
- Resize and standardization
- Optional GrabCut segmentation (experimental)
- Graceful failure handling (doesn't block pipeline)
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
class RootProcessingResult:
    """Result of root image processing."""
    success: bool
    segmentation_attempted: bool
    segmentation_success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RootPreprocessor:
    """
    Root image preprocessor with optional segmentation.
    
    Phase 1 baseline: focus on resize/standardization, with
    experimental GrabCut segmentation that gracefully fails.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        grabcut_enabled: bool = True,
        grabcut_margin: int = 15,
        grabcut_iterations: int = 5,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
        continue_on_failure: bool = True
    ):
        """
        Initialize root preprocessor.
        
        Args:
            target_size: Target image dimensions (width, height)
            grabcut_enabled: Whether to attempt GrabCut segmentation
            grabcut_margin: Margin for GrabCut rectangle initialization
            grabcut_iterations: Number of GrabCut iterations
            min_mask_ratio: Minimum acceptable mask coverage
            max_mask_ratio: Maximum acceptable mask coverage
            continue_on_failure: Continue pipeline if segmentation fails
        """
        self.target_size = target_size
        self.grabcut_enabled = grabcut_enabled
        self.grabcut_margin = grabcut_margin
        self.grabcut_iterations = grabcut_iterations
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.continue_on_failure = continue_on_failure
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def segment_grabcut(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Attempt GrabCut segmentation.
        
        Note: This is experimental for root images and may have high
        failure rates depending on background conditions.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (mask or None, error message)
        """
        try:
            h, w = image.shape[:2]
            margin = self.grabcut_margin
            
            # Initialize rectangle
            rect = (margin, margin, w - 2*margin, h - 2*margin)
            
            if rect[2] <= 0 or rect[3] <= 0:
                return None, "Invalid rectangle dimensions"
            
            # GrabCut
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(
                image, mask, rect,
                bgd_model, fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT
            )
            
            # Convert to binary mask
            binary_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Check mask coverage
            mask_ratio = np.sum(binary_mask > 0) / (h * w)
            
            if mask_ratio < self.min_mask_ratio:
                return None, f"Mask too small: {mask_ratio:.1%}"
            if mask_ratio > self.max_mask_ratio:
                return None, f"Mask too large: {mask_ratio:.1%}"
            
            return binary_mask, ""
            
        except Exception as e:
            return None, str(e)
    
    def apply_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """Apply mask to image."""
        result = np.full_like(image, background)
        mask_bool = mask > 0
        result[mask_bool] = image[mask_bool]
        return result
    
    def process(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], RootProcessingResult]:
        """
        Process a root image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (resized_image, mask_or_None, segmented_or_None, result)
        """
        # Resize
        resized = self.resize(image)
        
        mask = None
        segmented = None
        result = RootProcessingResult(
            success=True,
            segmentation_attempted=False,
            segmentation_success=False
        )
        
        # Attempt segmentation if enabled
        if self.grabcut_enabled:
            result.segmentation_attempted = True
            mask, error = self.segment_grabcut(resized)
            
            if mask is not None:
                result.segmentation_success = True
                segmented = self.apply_mask(resized, mask)
            else:
                result.error_message = error
                logger.warning(f"Root segmentation failed: {error}")
                
                if not self.continue_on_failure:
                    result.success = False
        
        return resized, mask, segmented, result


def process_root_image(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    preprocessor: Optional[RootPreprocessor] = None
) -> RootProcessingResult:
    """
    Process a single root image and save outputs.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        preprocessor: RootPreprocessor instance
        
    Returns:
        RootProcessingResult
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if preprocessor is None:
        preprocessor = RootPreprocessor()
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return RootProcessingResult(
            success=False,
            segmentation_attempted=False,
            segmentation_success=False,
            error_message=f"Failed to load image: {image_path}"
        )
    
    # Process
    resized, mask, segmented, result = preprocessor.process(image)
    
    # Save outputs
    stem = image_path.stem
    
    # Always save resized
    resized_path = output_dir / f"{stem}_root_resized.png"
    cv2.imwrite(str(resized_path), resized)
    
    # Save mask and segmented if available
    if mask is not None:
        mask_path = output_dir / f"{stem}_root_mask.png"
        cv2.imwrite(str(mask_path), mask)
    
    if segmented is not None:
        segmented_path = output_dir / f"{stem}_root_segmented.png"
        cv2.imwrite(str(segmented_path), segmented)
    
    return result


def batch_process_roots(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    preprocessor: Optional[RootPreprocessor] = None,
    extensions: List[str] = None
) -> Dict:
    """
    Batch process all root images in a directory.
    
    Args:
        input_dir: Directory containing root images
        output_dir: Directory to save outputs
        preprocessor: RootPreprocessor instance
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
    
    if preprocessor is None:
        preprocessor = RootPreprocessor()
    
    # Find images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    stats = {
        'total': len(image_files),
        'processed': 0,
        'segmentation_attempted': 0,
        'segmentation_success': 0,
        'segmentation_failed': 0,
        'errors': []
    }
    
    for image_path in tqdm(image_files, desc="Processing root images"):
        result = process_root_image(image_path, output_dir, preprocessor)
        
        if result.success:
            stats['processed'] += 1
        
        if result.segmentation_attempted:
            stats['segmentation_attempted'] += 1
            if result.segmentation_success:
                stats['segmentation_success'] += 1
            else:
                stats['segmentation_failed'] += 1
                stats['errors'].append({
                    'image': image_path.name,
                    'error': result.error_message
                })
    
    # Save stats
    stats_path = output_dir / 'root_processing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(
        f"Root processing complete: {stats['processed']}/{stats['total']}, "
        f"segmentation: {stats['segmentation_success']}/{stats['segmentation_attempted']}"
    )
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Root image preprocessing")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--no-segment", action='store_true', help="Disable segmentation")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    preprocessor = RootPreprocessor(grabcut_enabled=not args.no_segment)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = process_root_image(input_path, args.output, preprocessor)
        print(f"Result: {result.to_dict()}")
    else:
        stats = batch_process_roots(input_path, args.output, preprocessor)
        print(f"Statistics: {stats}")
