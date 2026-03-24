"""
Logistic Regression classifier for PlantDisease.

Trains a multi-class Logistic Regression model on features extracted
by the PlantDisease feature extraction pipeline (Gabor texture,
CIELAB colour, and morphology features).

Logistic Regression is included because:
  - It is the most interpretable classifier — feature weights directly
    show which features drive each disease prediction.
  - Your LAB colour stats and morphology ratios are well-scaled
    continuous values that suit linear classification well.
  - It provides a strong interpretable baseline to compare against
    XGBoost and SVM.

Usage
-----
Run directly:
    python classify.py

Or import for use in a notebook or training script:
    from src.plantdisease.models.logistic_regression.classify import (
        train, predict
    )
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared utilities — one level up in models/
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import cross_validate, evaluate, load_feature_names, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH  = "exports/features.csv"
MODEL_SAVE_PATH = "models/exports/logistic_regression.pkl"
RESULTS_PATH   = "exports/logistic_regression_results.csv"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# Logistic Regression hyperparameters
C        = 1.0      # inverse regularisation strength; lower = stronger regularisation
MAX_ITER = 1000     # increase if convergence warnings appear
SOLVER   = "lbfgs"  # handles multi-class well; memory efficient


# =============================================================================
# Train
# =============================================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = C,
    max_iter: int = MAX_ITER,
    solver: str = SOLVER,
) -> Pipeline:
    """Train a Logistic Regression classifier with standard scaling.

    Features are scaled to zero mean and unit variance before fitting
    since Logistic Regression is sensitive to feature scale — your Gabor
    responses, LAB values, and morphology metrics all operate on very
    different numerical ranges.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (integer encoded).
    C : float
        Inverse of regularisation strength (default: 1.0).
    max_iter : int
        Maximum solver iterations (default: 1000).
    solver : str
        Optimisation algorithm (default: 'lbfgs').

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline: StandardScaler → LogisticRegression.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    logger.info("Training Logistic Regression...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    logger.info("Logistic Regression training complete.")
    return pipeline


# =============================================================================
# Predict
# =============================================================================

def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Run inference on new samples.

    Parameters
    ----------
    model : Pipeline
        Fitted Logistic Regression Pipeline.
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
    class_names: np.ndarray,
    top_n: int = 10,
) -> None:
    """Print the top features driving each disease class prediction.

    Logistic Regression assigns a weight to each feature per class.
    Positive weight = feature increases probability of that class;
    negative weight = decreases it. This reveals which of your three
    feature groups (Gabor texture, LAB colour, morphology) are most
    discriminative per disease.

    Parameters
    ----------
    model : Pipeline
        Fitted Pipeline containing a LogisticRegression step.
    feature_names : np.ndarray
        Feature name strings from the npz or DataFrame.
    class_names : np.ndarray
        Disease class name strings.
    top_n : int
        Number of top features to show per class (default: 10).
    """
    import pandas as pd

    lr = model.named_steps["classifier"]
    coef_df = pd.DataFrame(
        lr.coef_.T,
        index=feature_names,
        columns=class_names,
    )

    print("\n" + "=" * 70)
    print(f"Logistic Regression — Top {top_n} Features Per Class")
    print("=" * 70)
    for cls in class_names:
        top = coef_df[cls].abs().nlargest(top_n).index
        print(f"\n{cls}:")
        print(coef_df.loc[top, cls].to_string())


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
        File path for the saved model (default: MODEL_SAVE_PATH).
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
    results = evaluate(model, X_test, y_test, class_names, "Logistic Regression")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "Logistic Regression")

    # ---- Feature importance ----
    try:
        feature_names = load_feature_names(FEATURES_PATH)
        feature_importance(model, feature_names, class_names)
    except KeyError:
        logger.warning("feature_names not found in npz — skipping importance analysis.")

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)