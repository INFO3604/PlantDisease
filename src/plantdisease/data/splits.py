"""
Train/Val/Test split creation with stratification.

This module handles:
- Creating stratified splits for imbalanced datasets
- Organizing processed images into training-ready folder structure
- Computing class weights for handling imbalance
- Generating split manifests
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.plantdisease import config

logger = logging.getLogger(__name__)


def make_splits(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = 'label',
    random_state: int = 42,
    task_filter: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test splits from manifest.
    
    Args:
        manifest_path: Path to manifest CSV
        output_dir: Where to save split files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        stratify_by: Column to stratify by (usually 'label')
        random_state: Random seed for reproducibility
        task_filter: Filter to only include 'leaf' or 'root' task
    
    Returns:
        Dictionary with DataFrames for train, val, test
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(df)} entries")
    
    # Filter by task if specified
    if task_filter:
        df = df[df['task'] == task_filter].copy()
        logger.info(f"Filtered to {len(df)} entries for task '{task_filter}'")
    
    # Filter to valid images only
    df = df[df['processing_status'] == 'valid'].copy()
    logger.info(f"Filtered to {len(df)} valid images")
    
    if len(df) == 0:
        raise ValueError("No valid images found in manifest")
    
    # Get stratification column
    stratify_col = df[stratify_by]
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_col,
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df[stratify_by],
        random_state=random_state
    )
    
    # Update split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Save split files
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Save combined manifest with splits
    combined = pd.concat([train_df, val_df, test_df])
    combined.to_csv(output_dir / 'manifest_with_splits.csv', index=False)
    combined.to_parquet(output_dir / 'manifest_with_splits.parquet', index=False)
    
    # Log split statistics
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    # Save split statistics
    stats = compute_split_statistics(splits)
    with open(output_dir / 'split_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return splits


def compute_split_statistics(splits: Dict[str, pd.DataFrame]) -> Dict:
    """
    Compute statistics for each split.
    
    Args:
        splits: Dictionary of split DataFrames
    
    Returns:
        Statistics dictionary
    """
    stats = {}
    
    for split_name, df in splits.items():
        class_counts = df['label'].value_counts().to_dict()
        
        stats[split_name] = {
            'total': len(df),
            'class_distribution': class_counts,
            'classes': list(class_counts.keys()),
            'num_classes': len(class_counts)
        }
    
    return stats


def compute_class_weights(
    train_df: pd.DataFrame,
    method: str = 'balanced'
) -> Dict[str, float]:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        train_df: Training DataFrame
        method: Weighting method ('balanced', 'sqrt', 'log')
    
    Returns:
        Dictionary mapping label to weight
    """
    class_counts = train_df['label'].value_counts()
    total = len(train_df)
    num_classes = len(class_counts)
    
    weights = {}
    
    if method == 'balanced':
        # Inverse frequency weighting
        for label, count in class_counts.items():
            weights[label] = total / (num_classes * count)
    
    elif method == 'sqrt':
        # Square root dampening
        for label, count in class_counts.items():
            weights[label] = np.sqrt(total / (num_classes * count))
    
    elif method == 'log':
        # Logarithmic dampening
        for label, count in class_counts.items():
            weights[label] = np.log(total / count + 1)
    
    else:
        # Equal weights
        for label in class_counts.keys():
            weights[label] = 1.0
    
    # Normalize weights to sum to num_classes
    weight_sum = sum(weights.values())
    for label in weights:
        weights[label] = weights[label] * num_classes / weight_sum
    
    return weights


def organize_for_training(
    splits: Dict[str, pd.DataFrame],
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    task: str = 'leaf',
    copy_files: bool = True,
    include_masks: bool = True,
    include_segmented: bool = True
) -> Path:
    """
    Organize images into training-ready folder structure.
    
    Creates structure:
        output_dir/
            {task}/
                train/
                    {label}/
                        image1.png
                        ...
                val/
                    {label}/
                        ...
                test/
                    {label}/
                        ...
    
    Args:
        splits: Dictionary of split DataFrames
        source_dir: Directory containing processed images
        output_dir: Root output directory
        task: 'leaf' or 'root'
        copy_files: If True, copy files; if False, create symlinks
        include_masks: Include mask files
        include_segmented: Include segmented images
    
    Returns:
        Path to organized data directory
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    task_dir = output_dir / task
    
    for split_name, df in splits.items():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Organizing {split_name}"):
            label = row['label']
            
            # Create label directory
            label_dir = task_dir / split_name / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Find source files
            image_id = row['image_id']
            
            # List of files to copy/link
            file_patterns = [
                f"{image_id}*resized*.png",
                f"{image_id}*resized*.jpg",
            ]
            
            if include_segmented:
                file_patterns.append(f"{image_id}*segmented*.png")
            
            if include_masks:
                file_patterns.append(f"{image_id}*mask*.png")
            
            for pattern in file_patterns:
                for src_file in source_dir.rglob(pattern):
                    dst_file = label_dir / src_file.name
                    
                    if copy_files:
                        if not dst_file.exists():
                            shutil.copy2(src_file, dst_file)
                    else:
                        if not dst_file.exists():
                            dst_file.symlink_to(src_file)
    
    logger.info(f"Organized data structure at {task_dir}")
    return task_dir


def load_split(
    split_name: str,
    split_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load a dataset split (train/val/test).
    
    Args:
        split_name: 'train', 'val', or 'test'
        split_dir: Directory containing split files
    
    Returns:
        DataFrame with image paths and labels
    """
    if split_dir is None:
        split_dir = config.SPLITS_DIR
    
    split_dir = Path(split_dir)
    split_file = split_dir / f"{split_name}.csv"
    
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    df = pd.read_csv(split_file)
    logger.info(f"Loaded {split_name} split: {len(df)} samples")
    
    return df


def get_label_mapping(splits: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional label mapping.
    
    Args:
        splits: Dictionary of split DataFrames
    
    Returns:
        Tuple of (label_to_idx, idx_to_label)
    """
    # Collect all labels
    all_labels = set()
    for df in splits.values():
        all_labels.update(df['label'].unique())
    
    # Sort for consistency
    labels = sorted(all_labels)
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    return label_to_idx, idx_to_label


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--manifest", "-m", required=True, help="Path to manifest CSV")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--task", choices=['leaf', 'root'], help="Filter by task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    splits = make_splits(
        manifest_path=args.manifest,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        task_filter=args.task,
        random_state=args.seed
    )
    
    print(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
