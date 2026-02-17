"""
Leaf segmentation module using GrabCut (primary) and Otsu thresholding (fallback).

This module provides:
- GrabCut-based leaf segmentation with configurable parameters
- Otsu thresholding as fallback when GrabCut fails
- Morphological post-processing (open/close, largest component extraction)
- Quality control checks and failure handling
- Overlay preview generation for QC
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SegmentationMethod(Enum):
    """Segmentation method used."""
    GRABCUT = "grabcut"
    OTSU = "otsu"
    FAILED = "failed"


@dataclass
class SegmentationResult:
    """Result of leaf segmentation."""
    success: bool
    method: str
    mask: Optional[np.ndarray]
    segmented: Optional[np.ndarray]
    mask_ratio: float
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding numpy arrays)."""
        return {
            'success': self.success,
            'method': self.method,
            'mask_ratio': self.mask_ratio,
            'error_message': self.error_message
        }


class LeafSegmenter:
    """
    Leaf segmentation using GrabCut with Otsu fallback.
    
    Primary mode: GrabCut with rectangular initialization
    Fallback mode: Otsu global thresholding
    """
    
    def __init__(
        self,
        grabcut_margin: int = 10,
        grabcut_iterations: int = 5,
        morph_kernel_size: int = 5,
        morph_open_iterations: int = 2,
        morph_close_iterations: int = 2,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
        use_otsu_fallback: bool = True
    ):
        """
        Initialize leaf segmenter.
        
        Args:
            grabcut_margin: Pixels inset from image edges for GrabCut rect
            grabcut_iterations: Number of GrabCut iterations
            morph_kernel_size: Kernel size for morphological operations
            morph_open_iterations: Iterations for morphological opening
            morph_close_iterations: Iterations for morphological closing
            min_mask_ratio: Minimum acceptable mask area ratio
            max_mask_ratio: Maximum acceptable mask area ratio
            use_otsu_fallback: Whether to use Otsu as fallback
        """
        self.grabcut_margin = grabcut_margin
        self.grabcut_iterations = grabcut_iterations
        self.morph_kernel_size = morph_kernel_size
        self.morph_open_iterations = morph_open_iterations
        self.morph_close_iterations = morph_close_iterations
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.use_otsu_fallback = use_otsu_fallback
        
        # Morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
    
    def segment_grabcut(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Segment leaf using GrabCut algorithm.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple of (binary mask or None, error message)
        """
        try:
            h, w = image.shape[:2]
            
            # Initialize rectangle (inset by margin)
            margin = self.grabcut_margin
            rect = (margin, margin, w - 2*margin, h - 2*margin)
            
            # Validate rectangle
            if rect[2] <= 0 or rect[3] <= 0:
                return None, "Invalid rectangle dimensions"
            
            # Initialize mask and models
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Run GrabCut
            cv2.grabCut(
                image, mask, rect,
                bgd_model, fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT
            )
            
            # Create binary mask (foreground = definite + probable)
            binary_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
            
            return binary_mask, ""
            
        except Exception as e:
            logger.error(f"GrabCut error: {e}")
            return None, str(e)
    
    def segment_otsu(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Segment leaf using Otsu thresholding.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple of (binary mask or None, error message)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Otsu's thresholding
            _, binary_mask = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Invert if background is white (leaf should be foreground)
            # Check if border pixels are mostly white (background)
            border = np.concatenate([
                binary_mask[0, :],      # Top row
                binary_mask[-1, :],     # Bottom row
                binary_mask[:, 0],      # Left column
                binary_mask[:, -1]      # Right column
            ])
            
            if np.mean(border) > 127:
                # Border is mostly white, invert mask
                binary_mask = cv2.bitwise_not(binary_mask)
            
            return binary_mask, ""
            
        except Exception as e:
            logger.error(f"Otsu error: {e}")
            return None, str(e)
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up mask.
        
        Operations:
        1. Morphological opening to remove small noise
        2. Morphological closing to fill small holes
        3. Keep only the largest connected component
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned binary mask
        """
        # Morphological opening (erosion then dilation)
        opened = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.morph_kernel,
            iterations=self.morph_open_iterations
        )
        
        # Morphological closing (dilation then erosion)
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, self.morph_kernel,
            iterations=self.morph_close_iterations
        )
        
        # Keep only largest connected component
        cleaned = self._keep_largest_component(closed)
        
        return cleaned
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component in the mask."""
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels <= 1:
            return mask
        
        # Find largest component (excluding background at label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create mask with only largest component
        result = np.zeros_like(mask)
        result[labels == largest_label] = 255
        
        return result
    
    def compute_mask_ratio(self, mask: np.ndarray) -> float:
        """Compute ratio of foreground pixels to total pixels."""
        total_pixels = mask.shape[0] * mask.shape[1]
        foreground_pixels = np.sum(mask > 0)
        return foreground_pixels / total_pixels
    
    def check_quality(self, mask: np.ndarray) -> Tuple[bool, str]:
        """
        Check if segmentation quality is acceptable.
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (is_acceptable, error_message)
        """
        ratio = self.compute_mask_ratio(mask)
        
        if ratio < self.min_mask_ratio:
            return False, f"Mask too small: {ratio:.1%} < {self.min_mask_ratio:.1%}"
        
        if ratio > self.max_mask_ratio:
            return False, f"Mask too large: {ratio:.1%} > {self.max_mask_ratio:.1%}"
        
        return True, ""
    
    def apply_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Apply mask to image, setting background to specified color.
        
        Args:
            image: Input image in BGR format
            mask: Binary mask (255 = foreground)
            background_color: BGR color for background
            
        Returns:
            Masked image
        """
        # Create output with background color
        result = np.full_like(image, background_color)
        
        # Copy foreground pixels
        mask_bool = mask > 0
        result[mask_bool] = image[mask_bool]
        
        return result
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Create overlay image showing mask on original.
        
        Args:
            image: Original image in BGR format
            mask: Binary mask
            alpha: Overlay transparency
            color: BGR color for mask overlay
            
        Returns:
            Overlay image
        """
        # Create colored overlay
        overlay = image.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] + 
            alpha * np.array(color)
        ).astype(np.uint8)
        
        return overlay
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Run complete segmentation pipeline.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            SegmentationResult with mask and metadata
        """
        # Try GrabCut first
        mask, error = self.segment_grabcut(image)
        method = SegmentationMethod.GRABCUT
        
        if mask is not None:
            # Apply morphological cleanup
            mask = self.apply_morphology(mask)
            
            # Check quality
            is_acceptable, qc_error = self.check_quality(mask)
            
            if not is_acceptable:
                logger.warning(f"GrabCut failed QC: {qc_error}")
                mask = None
                error = qc_error
        
        # Fallback to Otsu if GrabCut failed
        if mask is None and self.use_otsu_fallback:
            logger.info("Falling back to Otsu thresholding")
            mask, error = self.segment_otsu(image)
            method = SegmentationMethod.OTSU
            
            if mask is not None:
                # Apply morphological cleanup
                mask = self.apply_morphology(mask)
                
                # Check quality
                is_acceptable, qc_error = self.check_quality(mask)
                
                if not is_acceptable:
                    logger.warning(f"Otsu failed QC: {qc_error}")
                    mask = None
                    error = qc_error
        
        # Create result
        if mask is None:
            return SegmentationResult(
                success=False,
                method=SegmentationMethod.FAILED.value,
                mask=None,
                segmented=None,
                mask_ratio=0.0,
                error_message=error
            )
        
        # Apply mask to create segmented image
        segmented = self.apply_mask(image, mask)
        mask_ratio = self.compute_mask_ratio(mask)
        
        return SegmentationResult(
            success=True,
            method=method.value,
            mask=mask,
            segmented=segmented,
            mask_ratio=mask_ratio
        )


def segment_image(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    segmenter: Optional[LeafSegmenter] = None,
    save_overlay: bool = True
) -> SegmentationResult:
    """
    Segment a single image and save outputs.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        segmenter: LeafSegmenter instance (creates default if None)
        save_overlay: Whether to save overlay preview
        
    Returns:
        SegmentationResult
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create segmenter if not provided
    if segmenter is None:
        segmenter = LeafSegmenter()
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return SegmentationResult(
            success=False,
            method=SegmentationMethod.FAILED.value,
            mask=None,
            segmented=None,
            mask_ratio=0.0,
            error_message=f"Failed to load image: {image_path}"
        )
    
    # Run segmentation
    result = segmenter.segment(image)
    
    # Define output paths
    stem = image_path.stem
    mask_path = output_dir / f"{stem}_leaf_mask.png"
    segmented_path = output_dir / f"{stem}_leaf_segmented.png"
    overlay_path = output_dir / f"{stem}_overlay_preview.png"
    
    if result.success:
        # Save mask
        cv2.imwrite(str(mask_path), result.mask)
        
        # Save segmented image
        cv2.imwrite(str(segmented_path), result.segmented)
        
        # Save overlay preview
        if save_overlay:
            overlay = segmenter.create_overlay(image, result.mask)
            cv2.imwrite(str(overlay_path), overlay)
        
        logger.info(f"Segmented {image_path.name} using {result.method}, mask ratio: {result.mask_ratio:.1%}")
    else:
        logger.warning(f"Failed to segment {image_path.name}: {result.error_message}")
    
    return result


def batch_segment(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    qc_failures_dir: Optional[Union[str, Path]] = None,
    segmenter: Optional[LeafSegmenter] = None,
    extensions: List[str] = None
) -> Dict:
    """
    Batch segment all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        qc_failures_dir: Directory to save failed segmentations
        segmenter: LeafSegmenter instance
        extensions: List of valid extensions
        
    Returns:
        Dictionary with statistics
    """
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if qc_failures_dir:
        qc_failures_dir = Path(qc_failures_dir)
        qc_failures_dir.mkdir(parents=True, exist_ok=True)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    if segmenter is None:
        segmenter = LeafSegmenter()
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    # Statistics
    stats = {
        'total': len(image_files),
        'success': 0,
        'failed': 0,
        'grabcut_count': 0,
        'otsu_count': 0,
        'failures': []
    }
    
    # Process images
    for image_path in tqdm(image_files, desc="Segmenting images"):
        result = segment_image(image_path, output_dir, segmenter)
        
        if result.success:
            stats['success'] += 1
            if result.method == 'grabcut':
                stats['grabcut_count'] += 1
            else:
                stats['otsu_count'] += 1
        else:
            stats['failed'] += 1
            stats['failures'].append({
                'image': image_path.name,
                'error': result.error_message
            })
            
            # Save failure for QC
            if qc_failures_dir:
                # Copy original image to failures directory
                image = cv2.imread(str(image_path))
                if image is not None:
                    failure_path = qc_failures_dir / f"{image_path.stem}_failed.png"
                    cv2.imwrite(str(failure_path), image)
                    
                    # Save error reason
                    error_path = qc_failures_dir / f"{image_path.stem}_error.txt"
                    with open(error_path, 'w') as f:
                        f.write(result.error_message)
    
    # Save statistics
    stats_path = output_dir / 'segmentation_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Batch segmentation complete: {stats['success']}/{stats['total']} successful")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Leaf segmentation using GrabCut")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--qc-dir", help="Directory for QC failures")
    parser.add_argument("--margin", type=int, default=10, help="GrabCut margin")
    parser.add_argument("--iterations", type=int, default=5, help="GrabCut iterations")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    segmenter = LeafSegmenter(
        grabcut_margin=args.margin,
        grabcut_iterations=args.iterations
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = segment_image(input_path, args.output, segmenter)
        print(f"Result: {result.to_dict()}")
    else:
        stats = batch_segment(input_path, args.output, args.qc_dir, segmenter)
        print(f"Statistics: {stats}")
