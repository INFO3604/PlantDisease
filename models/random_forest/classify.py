"""
Random Forest classifier for PlantDisease.

Trains a multi-class Random Forest model on features extracted by the
PlantDisease feature extraction pipeline (Gabor texture, CIELAB colour,
and morphology features).

Random Forest is well suited to this feature set because:
  - Gabor features are high dimensional (36 features at default settings)
    and Random Forest handles this naturally by only considering a random
    subset of features at each split, preventing any one feature group
    from dominating.
  - Morphology features like lesion count, circularity, and eccentricity
    create natural decision tree boundaries — a lesion count of 1 vs 20
    is a clean split that trees exploit directly.
  - It requires no feature scaling unlike SVM and Logistic Regression,
    so the different numerical ranges of Gabor responses, LAB values,
    and shape metrics do not cause problems.
  - Feature importance scores show exactly which of your 3 extractors
    contribute most to classification, useful for analysis.

Usage
-----
Run directly:
    python classify.py

Or import for use in a notebook or training script:
    from src.plantdisease.models.random_forest.classify import train, predict
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import cross_validate, evaluate, load_feature_names, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH   = "exports/features.csv"
MODEL_SAVE_PATH = "models/exports/random_forest.pkl"
RESULTS_PATH    = "exports/random_forest_results.csv"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# Random Forest hyperparameters
N_ESTIMATORS = 500    # number of trees; more = more stable but slower to train
MAX_DEPTH    = None   # None = grow trees fully; set a number to limit depth
MIN_SAMPLES_SPLIT = 2 # minimum samples required to split a node
N_JOBS       = -1     # use all available CPU cores


# =============================================================================
# Train
# =============================================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
    min_samples_split: int = MIN_SAMPLES_SPLIT,
) -> Pipeline:
    """Train a Random Forest classifier.

    No feature scaling is required — Random Forest is invariant to the
    scale and range of input features, so Gabor responses, LAB stats,
    and morphology metrics can all be fed in directly.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (integer encoded).
    n_estimators : int
        Number of trees in the forest (default: 500).
    max_depth : int or None
        Maximum depth of each tree. None grows fully (default: None).
    min_samples_split : int
        Minimum samples needed to split a node (default: 2).

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline containing RandomForestClassifier.
        Wrapped in a Pipeline for consistency with other classifiers.
    """
    pipeline = Pipeline([
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight="balanced",  # handles class imbalance automatically
        )),
    ])

    logger.info(f"Training Random Forest ({n_estimators} trees)...")
    pipeline.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return pipeline


# =============================================================================
# Predict
# =============================================================================

def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Run inference on new samples.

    Parameters
    ----------
    model : Pipeline
        Fitted Random Forest Pipeline.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Predicted integer class labels.
    """
    return model.predict(X)


# =============================================================================
# Feature Importance
# =============================================================================

def feature_importance(
    model: Pipeline,
    feature_names: np.ndarray,
    top_n: int = 20,
) -> None:
    """Print the top features by importance from the trained Random Forest.

    Random Forest computes feature importance by measuring how much each
    feature reduces impurity across all trees. This will show you which
    of your Gabor, colour, and morphology features are most discriminative
    for disease classification.

    Parameters
    ----------
    model : Pipeline
        Fitted Pipeline containing a RandomForestClassifier step.
    feature_names : np.ndarray
        Feature name strings from the npz or DataFrame.
    top_n : int
        Number of top features to display (default: 20).
    """
    import pandas as pd

    rf = model.named_steps["classifier"]
    importance_df = pd.DataFrame({
        "feature":   feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n" + "=" * 70)
    print(f"Random Forest — Top {top_n} Most Important Features")
    print("=" * 70)
    print(importance_df.head(top_n).to_string(index=False))

    # Group by extractor prefix for a summary
    print("\nImportance by Feature Group:")
    importance_df["group"] = importance_df["feature"].apply(
        lambda x: "gabor" if x.startswith("gabor")
        else "colour" if any(x.startswith(p) for p in ["disease_l", "disease_a", "disease_b", "leaf_l", "leaf_a", "leaf_b", "disease_ratio", "yellow_ratio", "brown_ratio"])
        else "morphology"
    )
    group_summary = importance_df.groupby("group")["importance"].sum().sort_values(ascending=False)
    print(group_summary.to_string())


# =============================================================================
# Save / Load
# =============================================================================

def save_model(model: Pipeline, path: str = MODEL_SAVE_PATH) -> None:
    """Save the trained model to disk.

    Parameters
    ----------
    model : Pipeline
        Fitted Pipeline to save.
    path : str
        File path for the saved model.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str = MODEL_SAVE_PATH) -> Pipeline:
    """Load a saved model from disk.

    Parameters
    ----------
    path : str
        File path of the saved model.

    Returns
    -------
    Pipeline
        Loaded fitted Pipeline.
    """
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ---- Load data ----
    X, y, class_names = load_features(FEATURES_PATH)

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test samples")

    # ---- Train ----
    model = train(X_train, y_train)

    # ---- Evaluate on test set ----
    results = evaluate(model, X_test, y_test, class_names, "Random Forest")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "Random Forest")

    # ---- Feature importance ----
    try:
        feature_names = load_feature_names(FEATURES_PATH)
        feature_importance(model, feature_names)
    except KeyError:
        logger.warning("feature_names not found — skipping importance analysis.")

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)