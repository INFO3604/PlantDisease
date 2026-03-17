"""
Random Forest ensemble classifier for plant disease detection.

Uses the 55-feature vector from extract_features (Gabor texture,
CIELAB colour statistics, severity ratios, and lesion morphology)
to classify disease types.
"""

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class RFEnsembleClassifier:
    """
    Random Forest ensemble classifier for plant disease detection.

    Uses scikit-learn RandomForestClassifier with feature scaling
    and label encoding.
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
        y_train: Union[np.ndarray, List[str]],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Union[np.ndarray, List[str]]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Train the classifier.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
            X_val: Validation features (logged but not used for early stopping)
            y_val: Validation labels
            feature_names: Names of features
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
        """Predict class labels (decoded strings)."""
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
        y_test: Union[np.ndarray, List[str]],
    ) -> Dict:
        """
        Evaluate classifier on a test/validation set.

        Returns:
            Dictionary with accuracy, precision, recall, f1 (weighted & macro),
            per-class report, confusion matrix, and class labels.
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

        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (Gini importance)."""
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


def train_rf_ensemble(
    train_features_path: Union[str, Path],
    val_features_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "models/rf_ensemble",
    **model_params,
) -> Tuple[RFEnsembleClassifier, Dict]:
    """
    Train Random Forest ensemble from feature files.

    Args:
        train_features_path: Path to training features NPZ file
            (keys: 'features', 'labels', optionally 'feature_names')
        val_features_path: Path to validation features NPZ file
        output_dir: Output directory for saved model and metrics
        **model_params: Forwarded to RFEnsembleClassifier constructor

    Returns:
        Tuple of (trained classifier, metrics dictionary)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_data = np.load(train_features_path, allow_pickle=True)
    X_train = train_data["features"]
    y_train = train_data["labels"]
    feature_names = (
        train_data["feature_names"].tolist() if "feature_names" in train_data else None
    )

    logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Load validation data
    X_val, y_val = None, None
    if val_features_path:
        val_data = np.load(val_features_path, allow_pickle=True)
        X_val = val_data["features"]
        y_val = val_data["labels"]
        logger.info(f"Validation data: {X_val.shape[0]} samples")

    # Create and train classifier
    classifier = RFEnsembleClassifier(**model_params)
    classifier.fit(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate
    if X_val is not None:
        metrics = classifier.evaluate(X_val, y_val)
        logger.info(f"Validation accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Validation F1 (weighted): {metrics['f1_weighted']:.4f}")
    else:
        metrics = classifier.evaluate(X_train, y_train)
        logger.info(f"Training accuracy: {metrics['accuracy']:.4f}")

    # Save model
    classifier.save(output_dir / "rf_model")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance = classifier.get_feature_importance()
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance_sorted, f, indent=2)

    return classifier, metrics
