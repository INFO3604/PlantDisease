#!/usr/bin/env python
"""
CLI tool for batch preprocessing of plant disease images.

This script provides a command-line interface for preprocessing images
using the full preprocessing pipeline: resizing, denoising, contrast
enhancement, shadow detection, leaf segmentation, and disease segmentation.

Usage examples:
    # Preprocess all images in a directory
    python scripts/preprocess_cli.py --input data/raw --output data/processed
    
    # Preprocess with specific options
    python scripts/preprocess_cli.py --input data/raw --output data/processed --img-size 224 --denoise bilateral
    
    # Process only PNG files with parallel processing
    python scripts/preprocess_cli.py --input data/raw --output data/processed --extensions .png --n-workers 4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease import config
from src.plantdisease.data.preprocess.resize_standardize import resize_image, standardize_image
from src.plantdisease.data.preprocess.denoise import denoise_bilateral, denoise_median, denoise_gaussian
from src.plantdisease.data.preprocess.contrast import enhance_contrast_clahe, enhance_contrast_histogram
from src.plantdisease.data.preprocess.segmentation import segment_image
from src.plantdisease.data.preprocess.disease import segment_disease_from_image
from src.plantdisease.utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Batch image preprocessor with parallel processing support."""
    
    def __init__(
        self,
        img_size: int = 224,
        denoise_method: str = 'bilateral',
        contrast_method: str = 'clahe',
        n_workers: int = 4,
        save_segmentation_masks: bool = False
    ):
        """
        Initialize preprocessor.
        
        Args:
            img_size: Target image size
            denoise_method: Denoising method ('bilateral', 'median', 'gaussian')
            contrast_method: Contrast enhancement method ('clahe', 'histogram')
            n_workers: Number of parallel workers
            save_segmentation_masks: Whether to save segmentation masks
        """
        self.img_size = img_size
        self.denoise_method = denoise_method
        self.contrast_method = contrast_method
        self.n_workers = n_workers
        self.save_segmentation_masks = save_segmentation_masks
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
    
    def preprocess_image(
        self,
        image_path: Path,
        output_dir: Path,
        save_mask: bool = False
    ) -> Tuple[bool, str]:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for processed image
            save_mask: Whether to save segmentation mask
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return False, f"Failed to read image: {image_path.name}"
            
            # Convert BGR to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Step 1: Resize
            resized = resize_image(image_rgb, target_size=(self.img_size, self.img_size))
            
            # Step 2: Denoise
            if self.denoise_method == 'bilateral':
                denoised = denoise_bilateral(resized)
            elif self.denoise_method == 'median':
                denoised = denoise_median(resized)
            elif self.denoise_method == 'gaussian':
                denoised = denoise_gaussian(resized)
            else:
                denoised = resized
            
            # Step 3: Enhance contrast
            if self.contrast_method == 'clahe':
                enhanced = enhance_contrast_clahe(denoised)
            elif self.contrast_method == 'histogram':
                enhanced = enhance_contrast_histogram(denoised)
            else:
                enhanced = denoised
            
            # Step 4: Leaf segmentation
            try:
                leaf_mask = segment_image(enhanced)
                if leaf_mask is not None:
                    segmented = enhanced * (leaf_mask[:, :, np.newaxis] / 255.0)
                else:
                    segmented = enhanced
            except Exception:
                logger.warning(f"Leaf segmentation failed for {image_path.name}, using enhanced image")
                segmented = enhanced
            
            # Step 5: Disease segmentation (optional)
            disease_mask = None
            try:
                disease_result = segment_disease_from_image(segmented.astype(np.uint8))
                if isinstance(disease_result, tuple):
                    disease_mask = disease_result[0]  # First element is usually the mask
                else:
                    disease_mask = disease_result
            except Exception:
                logger.debug(f"Disease segmentation failed for {image_path.name}")
            
            # Step 6: Standardize
            final_image = standardize_image(segmented.astype(np.uint8))
            
            # Save processed image
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / image_path.name
            
            # Convert RGB to BGR for cv2.imwrite
            final_image_bgr = (final_image * 255).astype(np.uint8)
            final_image_bgr = cv2.cvtColor(final_image_bgr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), final_image_bgr)
            
            # Save masks if requested
            if save_mask:
                mask_dir = output_dir / "masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                
                # Save leaf mask if available
                if leaf_mask is not None:
                    cv2.imwrite(str(mask_dir / f"{image_path.stem}_leaf_mask.png"), leaf_mask)
                
                # Save disease mask if available
                if disease_mask is not None:
                    cv2.imwrite(str(mask_dir / f"{image_path.stem}_disease_mask.png"), disease_mask)
            
            return True, f"Successfully processed: {image_path.name}"
        
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: List[str] = None
    ) -> dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for processed images
            extensions: List of image extensions to process
        
        Returns:
            Dictionary with processing statistics
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        extensions = [ext.lower() for ext in extensions]
        
        # Collect all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return self.stats
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    self.preprocess_image,
                    image_path,
                    output_dir,
                    self.save_segmentation_masks
                ): image_path for image_path in image_files
            }
            
            with tqdm(total=len(futures), desc="Preprocessing") as pbar:
                for future in as_completed(futures):
                    success, message = future.result()
                    self.stats['total_processed'] += 1
                    
                    if success:
                        self.stats['successful'] += 1
                        logger.debug(message)
                    else:
                        self.stats['failed'] += 1
                        self.stats['errors'].append(message)
                        logger.warning(message)
                    
                    pbar.update(1)
        
        return self.stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch preprocess plant disease images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/preprocess_cli.py --input data/raw --output data/processed
  
  # With custom parameters
  python scripts/preprocess_cli.py --input data/raw --output data/processed \\
      --img-size 256 --denoise median --n-workers 8
  
  # Save segmentation masks
  python scripts/preprocess_cli.py --input data/raw --output data/processed \\
      --save-masks --extensions .jpg .png
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory containing raw images'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for processed images'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Target image size (default: 224)'
    )
    parser.add_argument(
        '--denoise',
        choices=['bilateral', 'median', 'gaussian'],
        default='bilateral',
        help='Denoising method (default: bilateral)'
    )
    parser.add_argument(
        '--contrast',
        choices=['clahe', 'histogram'],
        default='clahe',
        help='Contrast enhancement method (default: clahe)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='Image extensions to process (default: .jpg .jpeg .png .bmp)'
    )
    parser.add_argument(
        '--n-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save segmentation masks'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("Plant Disease Image Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Image size: {args.img_size}x{args.img_size}")
    logger.info(f"Denoising method: {args.denoise}")
    logger.info(f"Contrast method: {args.contrast}")
    logger.info(f"Parallel workers: {args.n_workers}")
    logger.info(f"Save segmentation masks: {args.save_masks}")
    
    # Process images
    preprocessor = ImagePreprocessor(
        img_size=args.img_size,
        denoise_method=args.denoise,
        contrast_method=args.contrast,
        n_workers=args.n_workers,
        save_segmentation_masks=args.save_masks
    )
    
    stats = preprocessor.process_directory(
        args.input,
        args.output,
        args.extensions
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total processed: {stats['total_processed']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    
    if stats['errors']:
        logger.info("\nErrors:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            logger.info(f"  - {error}")
        if len(stats['errors']) > 10:
            logger.info(f"  ... and {len(stats['errors']) - 10} more")
    
    logger.info("=" * 60)
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
