#!/usr/bin/env python
"""Download PlantVillage dataset and train XGBoost model.

This script:
1. Downloads the PlantVillage dataset (Solanaceae crops subset)
2. Extracts features using the FeatureExtractor
3. Trains and evaluates an XGBoost classifier
"""
import os
import sys
import logging
import zipfile
import urllib.request
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_plantvillage_subset():
    """
    Download PlantVillage dataset.
    
    Uses the PlantVillage dataset from GitHub (color images subset).
    Focuses on Solanaceae crops: Tomato, Pepper, Potato.
    """
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if Kaggle is available and authenticated
    kaggle_success = False
    
    # Check for Kaggle credentials file first
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    
    if kaggle_json.exists() or kaggle_env:
        try:
            import kaggle
            logger.info("Kaggle API authenticated. Downloading PlantVillage dataset...")
            
            # Download from Kaggle
            kaggle.api.dataset_download_files(
                'emmarex/plantdisease',
                path=str(data_dir),
                unzip=True
            )
            
            # The dataset extracts to PlantVillage folder
            plantvillage_dir = data_dir / "PlantVillage"
            if plantvillage_dir.exists():
                # Move contents up one level for our structure
                for item in plantvillage_dir.iterdir():
                    if item.is_dir():
                        dest = data_dir / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                shutil.rmtree(plantvillage_dir, ignore_errors=True)
            
            kaggle_success = True
            
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")
    else:
        logger.info("Kaggle credentials not found, using alternative download...")
    
    if kaggle_success:
        return True
    
    # Alternative: Download a sample subset from a public source
    # Using the PlantVillage sample from GitHub
    sample_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_path = data_dir / "plantvillage.zip"
    
    try:
        logger.info(f"Downloading from {sample_url}...")
        logger.info("This may take several minutes depending on your connection...")
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\rDownloading: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(sample_url, zip_path, reporthook)
        print()  # New line after progress
        
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Find and organize the color images
        extracted_dir = data_dir / "PlantVillage-Dataset-master"
        color_dir = extracted_dir / "raw" / "color"
        
        if color_dir.exists():
            # Filter to Solanaceae crops only
            solanaceae_patterns = ['Tomato', 'Pepper', 'Potato']
            
            for class_dir in color_dir.iterdir():
                if class_dir.is_dir():
                    # Check if it's a Solanaceae crop
                    if any(pattern in class_dir.name for pattern in solanaceae_patterns):
                        dest = data_dir / class_dir.name
                        if not dest.exists():
                            shutil.copytree(class_dir, dest)
                            logger.info(f"Copied: {class_dir.name}")
            
            # Clean up
            shutil.rmtree(extracted_dir, ignore_errors=True)
        
        # Remove zip file
        zip_path.unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("\nPlease download manually:")
        logger.info("1. Go to https://www.kaggle.com/datasets/emmarex/plantdisease")
        logger.info("2. Download and extract to data/raw/")
        logger.info("3. Ensure structure is: data/raw/<class_name>/<images>")
        return False


def extract_features(data_dir: Path, output_path: Path, use_grayscale: bool = False):
    """Extract features from all images in the dataset."""
    from src.plantdisease.models.features import FeatureExtractor, extract_features_from_directory
    
    logger.info(f"Extracting features from {data_dir}...")
    
    extractor = FeatureExtractor(use_grayscale=use_grayscale)
    features, names, labels = extract_features_from_directory(data_dir, extractor)
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=features,
        labels=labels,
        image_names=names,
        feature_names=extractor.feature_names
    )
    
    logger.info(f"Saved features to {output_path}")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Classes: {np.unique(labels)}")
    
    return features, names, labels


def split_features(features_path: Path, train_path: Path, val_path: Path, test_path: Path, 
                   val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Split features into train/val/test sets."""
    data = np.load(features_path, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    names = data['image_names']
    feature_names = data['feature_names']
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, names_temp, names_test = train_test_split(
        features, labels, names,
        test_size=test_ratio,
        stratify=labels,
        random_state=42
    )
    
    # Second split: train/val
    val_size = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
        X_temp, y_temp, names_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=42
    )
    
    # Save splits
    for path, X, y, n in [
        (train_path, X_train, y_train, names_train),
        (val_path, X_val, y_val, names_val),
        (test_path, X_test, y_test, names_test)
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, features=X, labels=y, image_names=n, feature_names=feature_names)
    
    logger.info(f"Split sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    return train_path, val_path, test_path


def train_model(train_path: Path, val_path: Path, model_dir: Path):
    """Train XGBoost classifier."""
    from src.plantdisease.models.xgboost_baseline import XGBoostClassifier
    
    # Load data
    train_data = np.load(train_path, allow_pickle=True)
    X_train = train_data['features']
    y_train = train_data['labels']
    feature_names = train_data['feature_names'].tolist()
    
    val_data = np.load(val_path, allow_pickle=True)
    X_val = val_data['features']
    y_val = val_data['labels']
    
    logger.info(f"Training with {len(y_train)} samples, validating with {len(y_val)} samples")
    
    # Train classifier
    classifier = XGBoostClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    
    classifier.fit(X_train, y_train, X_val, y_val, feature_names)
    
    # Evaluate on validation set
    metrics = classifier.evaluate(X_val, y_val)
    
    logger.info("=" * 50)
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    logger.info("=" * 50)
    
    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_plantvillage"
    classifier.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return classifier, metrics


def evaluate_on_test(classifier, test_path: Path):
    """Final evaluation on test set."""
    test_data = np.load(test_path, allow_pickle=True)
    X_test = test_data['features']
    y_test = test_data['labels']
    
    metrics = classifier.evaluate(X_test, y_test)
    
    logger.info("=" * 50)
    logger.info("TEST SET Results (Final):")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    logger.info("=" * 50)
    
    # Print per-class metrics
    if 'classification_report' in metrics:
        logger.info("\nPer-class classification report:")
        print(metrics['classification_report'])
    
    return metrics


def main():
    """Main pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PlantVillage and train XGBoost")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if data exists")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale preprocessing")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    features_dir = Path("data/features")
    model_dir = Path("models/xgboost")
    
    # Step 1: Download dataset
    if not args.skip_download:
        existing_classes = list(data_dir.glob("*/")) if data_dir.exists() else []
        if len(existing_classes) < 3:
            logger.info("Step 1: Downloading PlantVillage dataset...")
            success = download_plantvillage_subset()
            if not success:
                logger.error("Failed to download dataset. Exiting.")
                return
        else:
            logger.info(f"Step 1: Found {len(existing_classes)} existing classes, skipping download")
    
    # Check we have data
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []
    if not class_dirs:
        logger.error(f"No data found in {data_dir}. Please ensure dataset is downloaded.")
        return
    
    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs[:5]]}...")
    
    # Step 2: Extract features
    logger.info("Step 2: Extracting features...")
    suffix = "_gray" if args.grayscale else ""
    features_path = features_dir / f"all_features{suffix}.npz"
    
    if not features_path.exists():
        extract_features(data_dir, features_path, use_grayscale=args.grayscale)
    else:
        logger.info(f"Features already exist at {features_path}, skipping extraction")
    
    # Step 3: Split data
    logger.info("Step 3: Splitting into train/val/test...")
    train_path = features_dir / f"train{suffix}.npz"
    val_path = features_dir / f"val{suffix}.npz"
    test_path = features_dir / f"test{suffix}.npz"
    
    if not train_path.exists():
        split_features(features_path, train_path, val_path, test_path)
    else:
        logger.info("Splits already exist, skipping")
    
    # Step 4: Train model
    logger.info("Step 4: Training XGBoost classifier...")
    classifier, val_metrics = train_model(train_path, val_path, model_dir)
    
    # Step 5: Evaluate on test set
    logger.info("Step 5: Final evaluation on test set...")
    test_metrics = evaluate_on_test(classifier, test_path)
    
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Model saved to: {model_dir}/xgboost_plantvillage.json")
    logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info("=" * 50)
    
    return test_metrics


if __name__ == "__main__":
    main()
