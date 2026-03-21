"""
Shared utilities for PlantDisease classification models.

Used by all classifiers (Logistic Regression, SVM, XGBoost, etc.).
Provides data loading, evaluation, cross-validation, and result saving.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_features(features_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted features from a .npz or .csv file.

    Parameters
    ----------
    features_path : str
        Path to the features file. Supports:
          - .npz : expects keys 'features', 'labels', 'feature_names'
          - .csv : expects a 'label' column; all other columns are features

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Encoded integer labels.
    class_names : np.ndarray
        Array of disease class name strings, indexed by label integer.
    """
    path = Path(features_path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        X = data["features"].astype(np.float32)
        raw_labels = data["labels"]

    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column.")
        raw_labels = df["label"].values

        feature_df = df.drop(columns=[c for c in ["label", "image_id"] if c in df.columns])
        numeric_feature_df = feature_df.select_dtypes(include=[np.number])

        dropped_cols = [c for c in feature_df.columns if c not in numeric_feature_df.columns]
        if dropped_cols:
            logger.warning("Dropped non-numeric CSV columns: %s", dropped_cols)

        if numeric_feature_df.shape[1] == 0:
            raise ValueError("No numeric feature columns found in CSV.")

        X = numeric_feature_df.values.astype(np.float32)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .npz or .csv")

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    class_names = le.classes_

    logger.info(
        f"Loaded {X.shape[0]} samples, {X.shape[1]} features, "
        f"{len(class_names)} classes: {list(class_names)}"
    )

    return X, y, class_names


def load_feature_names(features_path: str) -> np.ndarray:
    """Load feature names from a .npz file.

    Parameters
    ----------
    features_path : str
        Path to the .npz features file.

    Returns
    -------
    np.ndarray
        Array of feature name strings.
    """
    data = np.load(features_path, allow_pickle=True)
    if "feature_names" not in data:
        raise KeyError("'feature_names' key not found in npz file.")
    return data["feature_names"]


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
    model_name: str = "Model",
) -> Dict:
    """Evaluate a trained classifier and print a formatted report.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Trained classifier with a predict() method.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels (integer encoded).
    class_names : np.ndarray
        Disease class name strings for display.
    model_name : str
        Display name for the report header.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro_f1, per_class_report,
        and confusion_matrix.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report   = classification_report(y_test, y_pred, target_names=class_names)
    cm       = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"{model_name} — Evaluation Results")
    print("=" * 70)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nPer-class Report:")
    print(report)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print("=" * 70 + "\n")

    return {
        "model_name":       model_name,
        "accuracy":         accuracy,
        "macro_f1":         macro_f1,
        "per_class_report": report,
        "confusion_matrix": cm,
    }


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    n_splits: int = 10,
) -> np.ndarray:
    """Run stratified k-fold cross-validation and print mean accuracy.

    Uses 10-fold stratified split matching the sweet pepper paper methodology
    to ensure class distribution is maintained across all folds.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Unfitted or fitted sklearn-compatible model.
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Full label array.
    model_name : str
        Display name for the report.
    n_splits : int
        Number of folds (default: 10).

    Returns
    -------
    np.ndarray
        Array of per-fold accuracy scores.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"\n{model_name} — {n_splits}-Fold Cross-Validation")
    print(f"  Per-fold accuracy : {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean accuracy     : {scores.mean():.4f} (+/- {scores.std():.4f})")

    return scores


# =============================================================================
# Result Saving
# =============================================================================

def save_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to a CSV file.

    Parameters
    ----------
    results : dict
        Results dictionary from evaluate().
    output_path : str
        Path to save the CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        "model":    results["model_name"],
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
    }])
    df.to_csv(path, index=False)
    logger.info(f"Results saved to {path}")