"""
Random Forest ensemble classifier for plant disease detection.

Uses extracted feature vectors from features.csv
to classify disease types.
"""
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from utils import cross_validate, evaluate, load_features, save_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH = "data/processed/features.csv"
MODEL_SAVE_PATH = "models/rf_ensemble/rf_model"
RESULTS_JSON_PATH = "exports/random_forest_metrics.json"
RESULTS_CSV_PATH = "exports/random_forest_results.csv"
FEATURE_IMPORTANCE_PATH = "exports/random_forest_feature_importance.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42


class RFEnsembleClassifier:
    """
    Random Forest ensemble classifier for plant disease detection.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: Optional[Union[str, Dict]] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Union[np.ndarray, List[Union[str, int]]],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Union[np.ndarray, List[Union[str, int]]]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Train the classifier.
        """
        self.feature_names = feature_names

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.rf.fit(X_train_scaled, y_train_encoded)

        logger.info(
            f"Training complete. Trees: {self.rf.n_estimators}, "
            f"Classes: {len(self.label_encoder.classes_)}"
        )

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation F1 (weighted): {val_metrics['f1_weighted']:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.rf.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.rf.predict_proba(X_scaled)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, List[Union[str, int]]],
    ) -> Dict:
        """
        Evaluate classifier on a test/validation set.
        """
        y_pred = self.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["per_class"] = report

        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["class_labels"] = self.label_encoder.classes_.tolist()

        print("\n" + "=" * 70)
        print("Random Forest — Evaluation Results")
        print("=" * 70)
        print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro F1 : {metrics['f1_macro']:.4f}")
        print("\nPer-class Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=self.label_encoder.classes_, columns=self.label_encoder.classes_).to_string())
        print("=" * 70 + "\n")

        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = self.rf.feature_importances_
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        return {f"feature_{i}": imp for i, imp in enumerate(importance)}

    def save(self, path: Union[str, Path]):
        """Save model, scaler, and label encoder to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "rf": self.rf,
            "label_encoder_classes": self.label_encoder.classes_,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "feature_names": self.feature_names,
        }
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Model saved to {path.with_suffix('.pkl')}")

    def load(self, path: Union[str, Path]):
        """Load model from disk."""
        path = Path(path)
        with open(path.with_suffix(".pkl"), "rb") as f:
            state = pickle.load(f)

        self.rf = state["rf"]
        self.label_encoder.classes_ = state["label_encoder_classes"]
        self.scaler.mean_ = state["scaler_mean"]
        self.scaler.scale_ = state["scaler_scale"]
        self.feature_names = state["feature_names"]

        logger.info(f"Model loaded from {path.with_suffix('.pkl')}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ---- Load data from CSV using shared utils ----
    X, y, class_names = load_features(FEATURES_PATH)

    # ---- Get feature names directly from CSV ----
    df = pd.read_csv(FEATURES_PATH)
    feature_names = df.drop(columns=["label", "image_id"], errors="ignore").columns.tolist()

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test samples")

    # ---- Train ----
    classifier = RFEnsembleClassifier(random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train, feature_names=feature_names)

    # ---- Evaluate ----
    metrics = classifier.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    # ---- Cross-validation ----
    cross_validate(classifier.rf, classifier.scaler.fit_transform(X), y, "Random Forest")

    # ---- Feature importance ----
    importance = classifier.get_feature_importance()
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\nTop 10 Important Features:")
    for k, v in list(importance_sorted.items())[:10]:
        print(f"{k}: {v:.6f}")

    # ---- Save model ----
    classifier.save(MODEL_SAVE_PATH)

    # ---- Save metrics JSON ----
    Path(RESULTS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Save summary CSV ----
    summary_df = pd.DataFrame([{
        "model": "Random Forest",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["f1_macro"],
    }])
    summary_df.to_csv(RESULTS_CSV_PATH, index=False)

    # ---- Save feature importance ----
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importance_sorted, f, indent=2)