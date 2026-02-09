"""Train/Val/Test split creation with stratification."""
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.plantdisease import config

logger = logging.getLogger(__name__)

def make_splits(data_dir=None, output_dir=None, train_ratio=0.8, val_ratio=0.1, 
                test_ratio=0.1, random_state=42):
    """
    Create train/val/test splits from processed data.
    
    Args:
        data_dir: Path to processed images
        output_dir: Where to save split files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with DataFrames for train, val, test
    """
    if data_dir is None:
        data_dir = config.PROCESSED_DATA_DIR
    if output_dir is None:
        output_dir = config.SPLITS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating splits from {data_dir}")
    logger.info(f"Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # TODO: Implement stratified split logic
    #   1. Load image list with labels
    #   2. Use sklearn.model_selection.train_test_split with stratify
    #   3. Save as CSV or TXT files to splits/
    
    splits = {
        "train": None,  # DataFrame with image paths and labels
        "val": None,
        "test": None,
    }
    
    return splits

def load_split(split_name, split_dir=None):
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
    
    split_file = split_dir / f"{split_name}.csv"
    
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    df = pd.read_csv(split_file)
    logger.info(f"Loaded {split_name} split: {len(df)} samples")
    
    return df
