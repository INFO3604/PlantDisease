"""
CatBoost classifier for PlantDisease.

Trains a multi-class CatBoost model on features extracted by the PlantDisease
feature extraction pipeline.

Usage
-----
python src/plantdisease/models/train_catboost.py
"""

import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from utils import cross_validate, evaluate, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH = "data/processed/features.csv"
MODEL_SAVE_PATH = "models/exports/catboost.pkl"
RESULTS_PATH = "exports/catboost_results.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# CatBoost hyperparameters
ITERATIONS = 800
LEARNING_RATE = 0.03
DEPTH = 10
L2_LEAF_REG = 1.0


# =============================================================================
# Train
# =============================================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    iterations: int = ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    depth: int = DEPTH,
    l2_leaf_reg: float = L2_LEAF_REG,
) -> CatBoostClassifier:
    """Train a CatBoost classifier."""
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        task_type="GPU",
        verbose=100,
    )

    logger.info("Training CatBoost...")
    model.fit(X_train, y_train)
    logger.info("CatBoost training complete.")
    return model


# =============================================================================
# Predict
# =============================================================================

def predict(model, X: np.ndarray) -> np.ndarray:
    """Run inference on new samples."""
    return model.predict(X).flatten().astype(int)


# =============================================================================
# Feature Importance
# =============================================================================

def feature_importance(
    model,
    feature_names: np.ndarray,
    top_n: int = 10,
) -> None:
    """Print top overall CatBoost features."""
    importance = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values(by="importance", ascending=False)

    print("\n" + "=" * 70)
    print(f"CatBoost — Top {top_n} Overall Features")
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
    results = evaluate(model, X_test, y_test, class_names, "CatBoost")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "CatBoost")

    # ---- Feature importance ----
    feature_importance(model, feature_names)

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)