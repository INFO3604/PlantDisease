"""
XGBoost classifier for PlantDisease.

Trains a multi-class XGBoost model on features extracted by the PlantDisease
feature extraction pipeline.

Usage
-----
python src/plantdisease/models/train_xgboost.py
"""

import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from utils import cross_validate, evaluate, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH = "data/processed/features.csv"
MODEL_SAVE_PATH = "models/exports/xgboost.pkl"
RESULTS_PATH = "exports/xgboost_results.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost hyperparameters
N_ESTIMATORS = 600
MAX_DEPTH = 10
LEARNING_RATE = 0.03
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


# =============================================================================
# Train
# =============================================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
    learning_rate: float = LEARNING_RATE,
    subsample: float = SUBSAMPLE,
    colsample_bytree: float = COLSAMPLE_BYTREE,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        device="cpu",
    )

    logger.info("Training XGBoost...")
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


# =============================================================================
# Predict
# =============================================================================

def predict(model, X: np.ndarray) -> np.ndarray:
    """Run inference on new samples."""
    return model.predict(X)


# =============================================================================
# Feature Importance
# =============================================================================

def feature_importance(
    model,
    feature_names: np.ndarray,
    top_n: int = 10,
) -> None:
    """Print top overall XGBoost features."""
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values(by="importance", ascending=False)

    print("\n" + "=" * 70)
    print(f"XGBoost — Top {top_n} Overall Features")
    print("=" * 70)
    print(importance_df.head(top_n).to_string(index=False))


# =============================================================================
# Save / Load
# =============================================================================

def save_model(model, path: str = MODEL_SAVE_PATH) -> None:
    """Save the trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str = MODEL_SAVE_PATH):
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
    results = evaluate(model, X_test, y_test, class_names, "XGBoost")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "XGBoost")

    # ---- Feature importance ----
    feature_importance(model, feature_names)

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)