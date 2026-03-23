"""
Support Vector Machine (SVM) classifier for PlantDisease.

Trains a multi-class SVM model on features extracted by the PlantDisease
feature extraction pipeline (Gabor texture, CIELAB colour, and morphology
features).

SVM is included because:
  - Your feature vector is compact and well-structured — exactly the input
    type SVM excels at.
  - Diseases like early blight and late blight share similar colour
    statistics and require non-linear decision boundaries to separate
    cleanly, which the RBF kernel handles well.
  - The cucurbit paper cites SVM with GLCM and LBP features achieving
    94% precision, directly comparable to your feature set.

Usage
-----
Run directly:
    python classify.py

Or import for use in a notebook or training script:
    from src.plantdisease.models.svm.classify import train, predict
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Shared utilities — one level up in models/
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import cross_validate, evaluate, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH   = "exports/features.csv"
MODEL_SAVE_PATH = "models/exports/svm.pkl"
RESULTS_PATH    = "exports/svm_results.csv"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# SVM hyperparameters
C      = 1.0     # regularisation; higher = tighter fit, higher overfitting risk
KERNEL = "rbf"   # RBF recommended for non-linear boundaries between disease classes
GAMMA  = "scale" # 1 / (n_features * X.var()); good default for mixed-scale features


# =============================================================================
# Train
# =============================================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = C,
    kernel: str = KERNEL,
    gamma: str = GAMMA,
) -> Pipeline:
    """Train an SVM classifier with standard scaling.

    Features are scaled to zero mean and unit variance before fitting —
    SVM is sensitive to feature scale and your Gabor responses, LAB values,
    and morphology metrics operate on very different numerical ranges.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (integer encoded).
    C : float
        Regularisation parameter (default: 1.0).
    kernel : str
        Kernel type: 'rbf' for non-linear boundaries (default),
        'linear' for a faster more interpretable model.
    gamma : str or float
        Kernel coefficient (default: 'scale').

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline: StandardScaler → SVC.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            decision_function_shape="ovr",
            random_state=RANDOM_STATE,
            probability=True,   # enables predict_proba for ROC curves if needed
        )),
    ])

    logger.info(f"Training SVM (kernel={kernel})...")
    pipeline.fit(X_train, y_train)
    logger.info("SVM training complete.")
    return pipeline


# =============================================================================
# Predict
# =============================================================================

def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Run inference on new samples.

    Parameters
    ----------
    model : Pipeline
        Fitted SVM Pipeline.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Predicted integer class labels.
    """
    return model.predict(X)


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
    results = evaluate(model, X_test, y_test, class_names, "SVM (RBF)")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "SVM (RBF)")

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)