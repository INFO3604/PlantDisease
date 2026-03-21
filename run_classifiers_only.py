"""
Run classifiers on pre-extracted features.

This script trains and evaluates Logistic Regression and SVM classifiers
on the features extracted from preprocessed images.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src and models to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from utils import load_features, evaluate
from logistic_regression.classify import train as train_lr
from svm.classify import train as train_svm
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Main
# ============================================================================

def main():
    """Run classifiers on pre-extracted features."""
    
    features_csv = PROJECT_ROOT / "exports" / "features.csv"
    
    logger.info("=" * 70)
    logger.info("PlantDisease Classifier Evaluation")
    logger.info("=" * 70)
    
    # Check if features file exists
    if not features_csv.exists():
        logger.error(f"Features file not found: {features_csv}")
        return
    
    # Fix the features CSV - remove image_id index if present
    logger.info(f"\n[STEP 1] Loading features from: {features_csv}")
    
    df = pd.read_csv(features_csv)
    
    # Check structure
    logger.info(f"Features shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns[:10])}...")
    
    # Make sure we have a 'label' column
    if 'label' not in df.columns:
        logger.error("No 'label' column found in features CSV")
        return
    
    # Extract labels and drop them from features
    y_labels = df['label'].values
    X = df.drop(columns=['label']).values
    
    # Try to drop image_id if it's in the features
    if 'image_id' in df.columns:
        X = df.drop(columns=['label', 'image_id']).values
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    class_names = le.classes_
    
    X = X.astype(np.float32)
    
    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Classes: {class_names}\n")
    
    # Train-test split
    logger.info("[STEP 2] Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples\n")
    
    results = {}
    
    # ---- Logistic Regression ----
    logger.info("=" * 70)
    logger.info("[STEP 3a] Training Logistic Regression...")
    logger.info("=" * 70)
    
    lr_model = train_lr(X_train, y_train)
    lr_results = evaluate(lr_model, X_test, y_test, class_names, model_name="Logistic Regression")
    results["logistic_regression"] = lr_results
    
    # ---- SVM ----
    logger.info("\n" + "=" * 70)
    logger.info("[STEP 3b] Training Support Vector Machine (SVM)...")
    logger.info("=" * 70)
    
    svm_model = train_svm(X_train, y_train)
    svm_results = evaluate(svm_model, X_test, y_test, class_names, model_name="SVM")
    results["svm"] = svm_results
    
    # ---- Compare Results ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Logistic Regression vs SVM")
    logger.info("=" * 70)
    
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": f"{model_results['accuracy'] * 100:.2f}%",
            "Macro F1": f"{model_results['macro_f1']:.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Determine winner
    accuracies = {
        name: res["accuracy"] for name, res in results.items()
    }
    best_model = max(accuracies, key=accuracies.get)
    logger.info(
        f"\n🏆 Best Model: {best_model.replace('_', ' ').title()} "
        f"(Accuracy: {accuracies[best_model] * 100:.2f}%)"
    )
    
    logger.info("=" * 70)
    logger.info("Classifier Evaluation Complete!")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
