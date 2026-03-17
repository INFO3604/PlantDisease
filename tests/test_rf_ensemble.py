"""
Test suite for Random Forest ensemble classifier.

Tests:
1. RFEnsembleClassifier training and prediction
2. Evaluation metrics
3. Model save/load round-trip
4. Feature importance
"""

import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease.models.rf_ensemble import RFEnsembleClassifier


@pytest.fixture
def dummy_data():
    """Create small synthetic dataset (3 classes, 55 features)."""
    rng = np.random.RandomState(42)
    n_samples = 120
    n_features = 55
    X = rng.randn(n_samples, n_features)
    y = np.array(["healthy", "yellow_spot", "brown_rot"] * (n_samples // 3))
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


class TestRFEnsembleClassifier:
    """Tests for the Random Forest ensemble classifier."""

    def test_fit_and_predict(self, dummy_data):
        X, y, feature_names = dummy_data
        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset(set(y))

    def test_predict_proba(self, dummy_data):
        X, y, _ = dummy_data
        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate_returns_metrics(self, dummy_data):
        X, y, _ = dummy_data
        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        metrics = clf.evaluate(X, y)
        for key in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

        assert "confusion_matrix" in metrics
        assert "class_labels" in metrics

    def test_feature_importance(self, dummy_data):
        X, y, feature_names = dummy_data
        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y, feature_names=feature_names)

        importance = clf.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())

    def test_save_and_load(self, dummy_data):
        X, y, feature_names = dummy_data
        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y, feature_names=feature_names)
        original_preds = clf.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rf_model"
            clf.save(model_path)

            clf2 = RFEnsembleClassifier()
            clf2.load(model_path)
            loaded_preds = clf2.predict(X)

            np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_validation_set(self, dummy_data):
        X, y, feature_names = dummy_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        preds = clf.predict(X_val)
        assert preds.shape == (len(X_val),)
