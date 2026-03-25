"""
Test suite for all six traditional classifiers.

Validates that every classifier can train on synthetic data,
make predictions, and produce sensible output shapes.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease.models.train_rf import RFEnsembleClassifier
from src.plantdisease.models import train_svm, train_logistic_regression, train_knn

# XGBoost and CatBoost are optional heavy dependencies
xgboost = pytest.importorskip("xgboost")
catboost = pytest.importorskip("catboost")
from src.plantdisease.models import train_xgboost, train_catboost


@pytest.fixture
def synth_data():
    """Small synthetic dataset: 90 samples, 55 features, 3 classes."""
    rng = np.random.RandomState(42)
    n = 90
    X = rng.randn(n, 55).astype(np.float32)
    y = np.array([0, 1, 2] * (n // 3))
    return X[:70], X[70:], y[:70], y[70:]


class TestRandomForest:
    def test_train_predict(self, synth_data):
        X_tr, X_te, y_tr, y_te = synth_data
        clf = RFEnsembleClassifier(n_estimators=10, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert preds.shape == (len(X_te),)


class TestSVM:
    def test_train_predict(self, synth_data):
        X_tr, X_te, y_tr, y_te = synth_data
        model = train_svm.train(X_tr, y_tr)
        preds = train_svm.predict(model, X_te)
        assert preds.shape == (len(X_te),)


class TestLogisticRegression:
    def test_train_predict(self, synth_data):
        X_tr, X_te, y_tr, y_te = synth_data
        model = train_logistic_regression.train(X_tr, y_tr)
        preds = train_logistic_regression.predict(model, X_te)
        assert preds.shape == (len(X_te),)


class TestXGBoost:
    def test_train_predict(self, synth_data):
        X_tr, X_te, y_tr, y_te = synth_data
        model = train_xgboost.train(X_tr, y_tr, n_estimators=10)
        preds = train_xgboost.predict(model, X_te)
        assert preds.shape == (len(X_te),)


class TestCatBoost:
    def test_train_predict(self, synth_data):
        X_tr, X_te, y_tr, y_te = synth_data
        model = train_catboost.train(X_tr, y_tr, iterations=10)
        preds = train_catboost.predict(model, X_te)
        assert preds.shape == (len(X_te),)


class TestKNN:
    def test_train_predict(self, synth_data):
        X_tr, X_te, y_tr, y_te = synth_data
        model = train_knn.train(X_tr, y_tr)
        preds = train_knn.predict(model, X_te)
        assert preds.shape == (len(X_te),)
