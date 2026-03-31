"""
Logistic Regression classifier for PlantDisease.

Trains a multi-class Logistic Regression model on features extracted
by the PlantDisease feature extraction pipeline.

Usage
-----
python src/plantdisease/models/train_logistic_regression.py
"""

import sys
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from utils import cross_validate, evaluate, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH = "data/processed/features.csv"
MODEL_SAVE_PATH = "models/exports/logistic_regression.pkl"
RESULTS_PATH = "exports/logistic_regression_results.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Logistic Regression hyperparameters
C = 10.0
MAX_ITER = 5000
SOLVER = "lbfgs"


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
    """Train a Logistic Regression classifier with standard scaling."""
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
    """Run inference on new samples."""
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
    """Print the top features driving each class prediction."""
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
    results = evaluate(model, X_test, y_test, class_names, "Logistic Regression")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "Logistic Regression")

    # ---- Feature importance ----
    feature_importance(model, feature_names, class_names)

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)