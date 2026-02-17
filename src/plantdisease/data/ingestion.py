"""
Data ingestion and validation module.

This module handles:
- Loading raw datasets organized by folders (class labels)
- Validating image files and removing corrupt images
- Creating manifests (CSV + Parquet) with metadata
- Supporting both leaf_task and root_task data
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion, validation, and manifest creation."""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(
        self,
        raw_data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        valid_extensions: Optional[List[str]] = None
    ):
        """
        Initialize data ingestion.
        
        Args:
            raw_data_dir: Root directory containing raw images organized by class
            output_dir: Directory to save manifests and processed data
            valid_extensions: List of valid image extensions
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if valid_extensions:
            self.valid_extensions = set(ext.lower() for ext in valid_extensions)
        else:
            self.valid_extensions = self.VALID_EXTENSIONS
        
        self.manifest_data = []
        self.stats = {
            'total_files': 0,
            'valid_images': 0,
            'corrupt_images': 0,
            'skipped_files': 0,
            'class_counts': {},
            'task_counts': {}
        }
    
    def validate_image(self, image_path: Path) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
        """
        Validate a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message, (width, height) or None)
        """
        try:
            # First try with PIL (more robust format detection)
            with Image.open(image_path) as img:
                img.verify()
            
            # Reopen to get dimensions (verify closes the file)
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Also verify OpenCV can read it
            cv_img = cv2.imread(str(image_path))
            if cv_img is None:
                return False, "OpenCV cannot read image", None
            
            # Check for minimum dimensions
            if width < 32 or height < 32:
                return False, f"Image too small: {width}x{height}", None
            
            return True, "", (width, height)
            
        except Exception as e:
            return False, str(e), None
    
    def compute_image_hash(self, image_path: Path) -> str:
        """Compute MD5 hash of image for deduplication."""
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def infer_task_from_path(self, image_path: Path) -> str:
        """
        Infer task (leaf or root) from path structure.
        
        Expected structure: raw_data_dir/leaf/ or raw_data_dir/root/
        If not found, defaults to 'leaf'.
        """
        path_parts = image_path.relative_to(self.raw_data_dir).parts
        
        for part in path_parts:
            part_lower = part.lower()
            if 'leaf' in part_lower or 'leaves' in part_lower:
                return 'leaf'
            elif 'root' in part_lower or 'roots' in part_lower:
                return 'root'
        
        return 'leaf'  # Default to leaf
    
    def infer_crop_from_path(self, image_path: Path) -> Optional[str]:
        """
        Try to infer crop type (tomato, pepper, potato, eggplant) from path.
        """
        path_str = str(image_path).lower()
        
        crops = ['tomato', 'pepper', 'potato', 'eggplant', 'bell_pepper']
        for crop in crops:
            if crop in path_str:
                return crop.replace('_', ' ').title()
        
        return None
    
    def infer_label_from_path(self, image_path: Path) -> str:
        """
        Infer class label from parent directory name.
        
        Expected structure: .../class_name/image.jpg
        """
        return image_path.parent.name
    
    def scan_directory(
        self,
        task_filter: Optional[str] = None,
        num_workers: int = 4,
        remove_corrupt: bool = True
    ) -> pd.DataFrame:
        """
        Scan directory and validate all images.
        
        Args:
            task_filter: Filter to only process 'leaf' or 'root' tasks
            num_workers: Number of parallel workers for validation
            remove_corrupt: If True, move corrupt images to quarantine folder
            
        Returns:
            DataFrame with manifest data
        """
        logger.info(f"Scanning directory: {self.raw_data_dir}")
        
        # Collect all image files
        image_files = []
        for ext in self.valid_extensions:
            image_files.extend(self.raw_data_dir.rglob(f'*{ext}'))
            image_files.extend(self.raw_data_dir.rglob(f'*{ext.upper()}'))
        
        image_files = list(set(image_files))  # Remove duplicates
        self.stats['total_files'] = len(image_files)
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Create quarantine directory for corrupt images
        quarantine_dir = self.output_dir / 'quarantine'
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        self.manifest_data = []
        corrupt_images = []
        
        def process_image(image_path: Path) -> Optional[Dict]:
            """Process a single image and return metadata."""
            # Infer metadata from path
            task = self.infer_task_from_path(image_path)
            
            # Apply task filter if specified
            if task_filter and task != task_filter:
                return None
            
            is_valid, error_msg, dimensions = self.validate_image(image_path)
            
            label = self.infer_label_from_path(image_path)
            crop = self.infer_crop_from_path(image_path)
            
            record = {
                'image_id': image_path.stem,
                'original_path': str(image_path.relative_to(self.raw_data_dir)),
                'absolute_path': str(image_path),
                'task': task,
                'crop': crop if crop else '',
                'label': label,
                'split': '',  # To be filled by splitting module
                'width': dimensions[0] if dimensions else 0,
                'height': dimensions[1] if dimensions else 0,
                'processing_status': 'valid' if is_valid else 'corrupt',
                'notes': error_msg if error_msg else '',
                'file_hash': ''  # Computed separately if needed
            }
            
            return record
        
        # Process with progress bar
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image, path): path for path in image_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating images"):
                try:
                    result = future.result()
                    if result is not None:
                        self.manifest_data.append(result)
                        
                        if result['processing_status'] == 'valid':
                            self.stats['valid_images'] += 1
                            
                            # Update class counts
                            label = result['label']
                            task = result['task']
                            
                            if label not in self.stats['class_counts']:
                                self.stats['class_counts'][label] = 0
                            self.stats['class_counts'][label] += 1
                            
                            if task not in self.stats['task_counts']:
                                self.stats['task_counts'][task] = 0
                            self.stats['task_counts'][task] += 1
                        else:
                            self.stats['corrupt_images'] += 1
                            corrupt_images.append(futures[future])
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
        
        # Handle corrupt images
        if remove_corrupt and corrupt_images:
            logger.warning(f"Moving {len(corrupt_images)} corrupt images to quarantine")
            for img_path in corrupt_images:
                try:
                    dest = quarantine_dir / img_path.name
                    img_path.rename(dest)
                except Exception as e:
                    logger.error(f"Failed to quarantine {img_path}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(self.manifest_data)
        
        # Sort by task, label, image_id
        if not df.empty:
            df = df.sort_values(['task', 'label', 'image_id']).reset_index(drop=True)
        
        return df
    
    def save_manifest(
        self,
        df: pd.DataFrame,
        manifest_name: str = 'manifest'
    ) -> Tuple[Path, Path]:
        """
        Save manifest to CSV and Parquet formats.
        
        Args:
            df: DataFrame with manifest data
            manifest_name: Base name for output files
            
        Returns:
            Tuple of (csv_path, parquet_path)
        """
        csv_path = self.output_dir / f'{manifest_name}.csv'
        parquet_path = self.output_dir / f'{manifest_name}.parquet'
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved manifest to {csv_path} and {parquet_path}")
        
        return csv_path, parquet_path
    
    def save_stats(self, stats_name: str = 'ingestion_stats') -> Path:
        """Save ingestion statistics to JSON."""
        stats_path = self.output_dir / f'{stats_name}.json'
        
        # Add timestamp
        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['raw_data_dir'] = str(self.raw_data_dir)
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved stats to {stats_path}")
        return stats_path
    
    def get_label_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Create label to integer mapping.
        
        Args:
            df: Manifest DataFrame
            
        Returns:
            Dictionary mapping label names to integers
        """
        labels = sorted(df['label'].unique())
        return {label: idx for idx, label in enumerate(labels)}
    
    def print_summary(self):
        """Print summary of ingestion results."""
        print("\n" + "="*50)
        print("DATA INGESTION SUMMARY")
        print("="*50)
        print(f"Total files scanned: {self.stats['total_files']}")
        print(f"Valid images: {self.stats['valid_images']}")
        print(f"Corrupt images: {self.stats['corrupt_images']}")
        print(f"\nTask distribution:")
        for task, count in self.stats['task_counts'].items():
            print(f"  {task}: {count}")
        print(f"\nClass distribution:")
        for label, count in sorted(self.stats['class_counts'].items()):
            print(f"  {label}: {count}")
        print("="*50 + "\n")


def run_ingestion(
    raw_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    task_filter: Optional[str] = None,
    remove_corrupt: bool = True
) -> pd.DataFrame:
    """
    Run the complete data ingestion pipeline.
    
    Args:
        raw_data_dir: Directory containing raw images
        output_dir: Directory to save manifests
        task_filter: Optional filter for 'leaf' or 'root' task
        remove_corrupt: Whether to quarantine corrupt images
        
    Returns:
        Manifest DataFrame
    """
    ingestion = DataIngestion(raw_data_dir, output_dir)
    df = ingestion.scan_directory(task_filter=task_filter, remove_corrupt=remove_corrupt)
    
    ingestion.save_manifest(df, manifest_name='manifest')
    ingestion.save_stats()
    ingestion.print_summary()
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data ingestion and validation")
    parser.add_argument("--input", "-i", required=True, help="Raw data directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--task", choices=['leaf', 'root'], help="Filter by task type")
    parser.add_argument("--keep-corrupt", action='store_true', help="Don't quarantine corrupt images")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    run_ingestion(
        raw_data_dir=args.input,
        output_dir=args.output,
        task_filter=args.task,
        remove_corrupt=not args.keep_corrupt
    )
