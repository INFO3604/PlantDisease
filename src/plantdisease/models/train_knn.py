"""
K-Nearest Neighbours (KNN) classifier for PlantDisease.

Trains a KNN model on features extracted by the PlantDisease
feature extraction pipeline.

Usage
-----
python src/plantdisease/models/train_knn.py
"""

import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
MODEL_SAVE_PATH = "models/exports/knn.pkl"
RESULTS_PATH = "exports/knn_results.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

N_NEIGHBORS = 6
WEIGHTS = "distance"
METRIC = "manhattan"
RF_IMPORTANCE_POWER = 2.6


# =============================================================================
# RF-importance feature weighter (learned metric for KNN)
# =============================================================================

class RFImportanceWeighter(BaseEstimator, TransformerMixin):
    """Scale each feature by RF importance^power so KNN uses a learned metric."""

    def __init__(self, n_estimators=1000, power=RF_IMPORTANCE_POWER, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.power = power
        self.random_state = random_state

    def fit(self, X, y=None):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        self.weights_ = rf.feature_importances_ ** self.power
        return self

    def transform(self, X):
        return X * self.weights_


# =============================================================================
# Train
# =============================================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = N_NEIGHBORS,
    weights: str = WEIGHTS,
    metric: str = METRIC,
) -> Pipeline:
    """Train a KNN classifier with scaling and RF-importance feature weighting."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf_weighter", RFImportanceWeighter()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1,
        )),
    ])

    logger.info(f"Training KNN (k={n_neighbors}, weights={weights})...")
    pipeline.fit(X_train, y_train)
    logger.info("KNN training complete.")
    return pipeline


# =============================================================================
# Predict
# =============================================================================

def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Run inference on new samples."""
    return model.predict(X)


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
    results = evaluate(model, X_test, y_test, class_names, "KNN")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # ---- Cross-validation ----
    cross_validate(model, X, y, "KNN")

    # ---- Save model and results ----
    save_model(model)
    save_results(results, RESULTS_PATH)
