"""
XGBoost baseline classifier for plant disease detection.

This baseline uses traditional ML features:
- HSV color histograms
- LBP texture features
- GLCM texture features

Provides a non-deep-learning baseline for comparison.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    XGBoost classifier for plant disease detection.
    
    Uses traditional features extracted from images.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_gpu: bool = False,
        class_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns per tree
            random_state: Random seed
            use_gpu: Use GPU acceleration if available
            class_weights: Dictionary mapping labels to weights
        """
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("xgboost required. Install with: pip install xgboost")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob'
        }
        
        if use_gpu:
            self.params['tree_method'] = 'gpu_hist'
            self.params['predictor'] = 'gpu_predictor'
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.class_weights = class_weights
        self.feature_names = None
    
    def _compute_sample_weights(
        self,
        labels: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute sample weights from class weights."""
        if self.class_weights is None:
            return None
        
        weights = np.array([
            self.class_weights.get(label, 1.0) for label in labels
        ])
        return weights
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: Union[np.ndarray, List[str]],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Union[np.ndarray, List[str]]] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 10
    ):
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features
            early_stopping_rounds: Early stopping patience
        """
        self.feature_names = feature_names
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Get number of classes
        num_classes = len(self.label_encoder.classes_)
        self.params['num_class'] = num_classes
        
        # Compute sample weights
        sample_weights = self._compute_sample_weights(y_train)
        
        # Add early stopping to params if validation set provided
        model_params = self.params.copy()
        if X_val is not None and y_val is not None:
            model_params['early_stopping_rounds'] = early_stopping_rounds
        
        # Create model
        self.model = self.xgb.XGBClassifier(**model_params)
        
        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val_encoded)]
        
        # Train
        fit_params = {
            'sample_weight': sample_weights,
            'verbose': True
        }
        
        if eval_set:
            fit_params['eval_set'] = eval_set
        
        self.model.fit(X_train_scaled, y_train_encoded, **fit_params)
        
        logger.info(f"Training complete. Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels (decoded)
        """
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, List[str]]
    ) -> Dict:
        """
        Evaluate classifier on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
        }
        
        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['per_class'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['class_labels'] = self.label_encoder.classes_.tolist()
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def save(self, path: Union[str, Path]):
        """
        Save model to disk.
        
        Args:
            path: Output path (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path.with_suffix('.json')))
        
        # Save additional components
        components = {
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'params': self.params,
            'feature_names': self.feature_names
        }
        
        with open(path.with_suffix('.components.json'), 'w') as f:
            json.dump(components, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """
        Load model from disk.
        
        Args:
            path: Model path (without extension)
        """
        path = Path(path)
        
        # Load XGBoost model
        self.model = self.xgb.XGBClassifier()
        self.model.load_model(str(path.with_suffix('.json')))
        
        # Load components
        with open(path.with_suffix('.components.json'), 'r') as f:
            components = json.load(f)
        
        self.label_encoder.classes_ = np.array(components['label_encoder_classes'])
        self.scaler.mean_ = np.array(components['scaler_mean'])
        self.scaler.scale_ = np.array(components['scaler_scale'])
        self.params = components['params']
        self.feature_names = components['feature_names']
        
        logger.info(f"Model loaded from {path}")


def train_xgboost(
    train_features_path: Union[str, Path],
    val_features_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = 'models/xgboost',
    class_weights: Optional[Dict[str, float]] = None,
    **model_params
) -> Tuple[XGBoostClassifier, Dict]:
    """
    Train XGBoost classifier from feature files.
    
    Args:
        train_features_path: Path to training features NPZ file
        val_features_path: Path to validation features NPZ file
        output_dir: Output directory for model
        class_weights: Class weights for imbalanced data
        **model_params: Additional model parameters
    
    Returns:
        Tuple of (trained classifier, metrics dictionary)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    train_data = np.load(train_features_path, allow_pickle=True)
    X_train = train_data['features']
    y_train = train_data['labels']
    feature_names = train_data['feature_names'].tolist() if 'feature_names' in train_data else None
    
    logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Load validation data
    X_val, y_val = None, None
    if val_features_path:
        val_data = np.load(val_features_path, allow_pickle=True)
        X_val = val_data['features']
        y_val = val_data['labels']
        logger.info(f"Validation data: {X_val.shape[0]} samples")
    
    # Create and train classifier
    classifier = XGBoostClassifier(class_weights=class_weights, **model_params)
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
    classifier.save(output_dir / 'xgboost_model')
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        json_metrics['confusion_matrix'] = metrics['confusion_matrix']
        json.dump(json_metrics, f, indent=2)
    
    # Save feature importance
    importance = classifier.get_feature_importance()
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_dir / 'feature_importance.json', 'w') as f:
        json.dump(dict(importance_sorted), f, indent=2)
    
    return classifier, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost classifier")
    parser.add_argument("--train", required=True, help="Training features NPZ")
    parser.add_argument("--val", help="Validation features NPZ")
    parser.add_argument("--output", "-o", default="models/xgboost", help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    classifier, metrics = train_xgboost(
        train_features_path=args.train,
        val_features_path=args.val,
        output_dir=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    
    print(f"\nFinal metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
