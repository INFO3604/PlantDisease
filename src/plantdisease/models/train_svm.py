"""
Support Vector Machine (SVM) classifier for PlantDisease.

Trains a multi-class SVM model on features extracted by the PlantDisease
feature extraction pipeline.

Usage
-----
python src/plantdisease/models/train_svm.py
"""

import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent))
from utils import cross_validate, evaluate, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH = "data/processed/features.csv"
MODEL_SAVE_PATH = "models/exports/svm.pkl"
RESULTS_PATH = "exports/svm_results.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Changed to linear so feature importance is available
C = 1.0
KERNEL = "linear"
GAMMA = "scale"


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
    """Train an SVM classifier with standard scaling."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            decision_function_shape="ovr",
            random_state=RANDOM_STATE,
            probability=True,
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
    """Run inference on new samples."""
    return model.predict(X)


# =============================================================================
# Feature Importance
# =============================================================================

def feature_importance(
    model: Pipeline,
    feature_names: np.ndarray,
    top_n: int = 10,
) -> None:
    """
    Print top overall SVM features.

    For multiclass SVM, sklearn uses one-vs-one internally, so coef_ has shape:
    (n_classifiers, n_features) instead of (n_classes, n_features).

    To get a single interpretable ranking, we average the absolute coefficient
    magnitudes across all binary classifiers.
    """
    svm = model.named_steps["classifier"]

    if not hasattr(svm, "coef_"):
        logger.warning("This SVM kernel does not expose coefficients.")
        return

    coef = svm.coef_  # shape: (n_binary_classifiers, n_features)
    avg_coef = np.mean(np.abs(coef), axis=0)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_coef,
    }).sort_values(by="importance", ascending=False)

    print("\n" + "=" * 70)
    print(f"SVM (Linear) — Top {top_n} Overall Features")
    print("=" * 70)
    print(coef_df.head(top_n).to_string(index=False))


# =============================================================================
# Save / Load
# =============================================================================

def save_model(model: Pipeline, path: str = MODEL_SAVE_PATH) -> None:
    """Save the trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str = MODEL_SAVE_PATH) -> Pipeline:
    """Load a saved model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ---- Load data ----
    X, y, class_names = load_features(FEATURES_PATH)

    # ---- Get feature names directly from CSV ----
    df = pd.read_csv(FEATURES_PATH)
    feature_names = df.drop(columns=["label", "image_id"], errors="ignore").columns.values

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
    results = evaluate(model, X_test, y_test, class_names, "SVM (Linear)")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "SVM (Linear)")

    # ---- Feature importance ----
    feature_importance(model, feature_names)

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)