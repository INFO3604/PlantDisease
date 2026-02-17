"""
Test suite for Task 1: Feature Extraction & XGBoost Baseline Model

Tests:
1. Feature extraction (HSV, LBP, GLCM)
2. Grayscale preprocessing option
3. XGBoost classifier training and evaluation
4. Model save/load functionality
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plantdisease.models.features import (
    HSVHistogramExtractor,
    LBPExtractor, 
    GLCMExtractor,
    FeatureExtractor,
    extract_features_from_directory
)
from src.plantdisease.models.xgboost_baseline import XGBoostClassifier


class TestHSVHistogramExtractor:
    """Tests for HSV histogram feature extraction."""
    
    def test_extract_features_shape(self):
        """Test that extracted features have correct shape."""
        extractor = HSVHistogramExtractor(h_bins=32, s_bins=32, v_bins=32)
        
        # Create dummy image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        features = extractor.extract(image)
        
        assert features.shape == (96,), f"Expected (96,), got {features.shape}"
        assert extractor.feature_dim == 96
    
    def test_extract_with_mask(self):
        """Test feature extraction with mask."""
        extractor = HSVHistogramExtractor()
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # Only center region
        
        features = extractor.extract(image, mask)
        
        assert features.shape[0] == extractor.feature_dim
    
    def test_normalization(self):
        """Test that histogram is normalized."""
        extractor = HSVHistogramExtractor(normalize=True)
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = extractor.extract(image)
        
        # Sum of each channel's histogram bins should be close to 1/3
        # (since we have 3 concatenated histograms)
        assert np.isclose(np.sum(features), 1.0, atol=0.01)


class TestLBPExtractor:
    """Tests for LBP texture feature extraction."""
    
    def test_extract_features_shape(self):
        """Test LBP feature shape."""
        extractor = LBPExtractor(radius=3, n_points=24, n_bins=26)
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = extractor.extract(image)
        
        assert features.shape == (26,), f"Expected (26,), got {features.shape}"
    
    def test_grayscale_conversion(self):
        """Test that color images are properly converted to grayscale."""
        extractor = LBPExtractor()
        
        # Color image
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features_color = extractor.extract(color_image)
        
        # Already grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        features_gray = extractor.extract(gray_3ch)
        
        # Both should produce valid features
        assert features_color.shape == features_gray.shape


class TestGLCMExtractor:
    """Tests for GLCM texture feature extraction."""
    
    def test_extract_features_shape(self):
        """Test GLCM feature shape."""
        extractor = GLCMExtractor(distances=[1, 2], angles=[0, np.pi/2], levels=32)
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = extractor.extract(image)
        
        # 6 properties * 2 distances * 2 angles = 24 features
        expected_dim = 6 * 2 * 2
        assert features.shape == (expected_dim,), f"Expected ({expected_dim},), got {features.shape}"


class TestFeatureExtractor:
    """Tests for combined feature extractor."""
    
    def test_extract_all_features(self):
        """Test combined feature extraction."""
        extractor = FeatureExtractor()
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = extractor.extract(image)
        
        assert features.shape[0] == extractor.feature_dim
        assert len(extractor.feature_names) == extractor.feature_dim
    
    def test_grayscale_option(self):
        """Test grayscale preprocessing option."""
        # Without grayscale
        extractor_color = FeatureExtractor(use_grayscale=False)
        
        # With grayscale
        extractor_gray = FeatureExtractor(use_grayscale=True, grayscale_method='luminosity')
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        features_color = extractor_color.extract(image)
        features_gray = extractor_gray.extract(image)
        
        # Both should produce same shape
        assert features_color.shape == features_gray.shape
        
        # But different values (since input is different)
        assert not np.allclose(features_color, features_gray)
    
    def test_different_grayscale_methods(self):
        """Test different grayscale conversion methods."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        methods = ['luminosity', 'average', 'lightness', 'opencv']
        features_list = []
        
        for method in methods:
            extractor = FeatureExtractor(use_grayscale=True, grayscale_method=method)
            features = extractor.extract(image)
            features_list.append(features)
        
        # All should have same shape
        for feats in features_list:
            assert feats.shape == features_list[0].shape


class TestXGBoostClassifier:
    """Tests for XGBoost baseline classifier."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 194
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.array(['healthy', 'chlorosis', 'necrosis'] * 33 + ['healthy'])
        
        X_val = np.random.randn(20, n_features)
        y_val = np.array(['healthy', 'chlorosis', 'necrosis', 'healthy'] * 5)
        
        return X_train, y_train, X_val, y_val
    
    def test_train_and_predict(self, dummy_data):
        """Test training and prediction."""
        X_train, y_train, X_val, y_val = dummy_data
        
        clf = XGBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train, X_val, y_val)
        
        predictions = clf.predict(X_val)
        
        assert len(predictions) == len(y_val)
        assert all(p in ['healthy', 'chlorosis', 'necrosis'] for p in predictions)
    
    def test_predict_proba(self, dummy_data):
        """Test probability prediction."""
        X_train, y_train, X_val, y_val = dummy_data
        
        clf = XGBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train)
        
        proba = clf.predict_proba(X_val)
        
        assert proba.shape == (len(X_val), 3)  # 3 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate_metrics(self, dummy_data):
        """Test evaluation metrics."""
        X_train, y_train, X_val, y_val = dummy_data
        
        clf = XGBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train)
        
        metrics = clf.evaluate(X_val, y_val)
        
        assert 'accuracy' in metrics
        assert 'f1_weighted' in metrics
        assert 'confusion_matrix' in metrics
        assert 'class_labels' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_weighted'] <= 1
    
    def test_feature_importance(self, dummy_data):
        """Test feature importance extraction."""
        X_train, y_train, X_val, y_val = dummy_data
        
        clf = XGBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train, feature_names=[f'feat_{i}' for i in range(194)])
        
        importance = clf.get_feature_importance()
        
        assert len(importance) == 194
        assert all(v >= 0 for v in importance.values())
    
    def test_save_and_load(self, dummy_data):
        """Test model save and load."""
        X_train, y_train, X_val, y_val = dummy_data
        
        clf = XGBoostClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train)
        
        original_predictions = clf.predict(X_val)
        
        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model'
            clf.save(model_path)
            
            # Load in new classifier
            clf2 = XGBoostClassifier()
            clf2.load(model_path)
            
            loaded_predictions = clf2.predict(X_val)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_class_weights(self, dummy_data):
        """Test class weighting for imbalanced data."""
        X_train, y_train, X_val, y_val = dummy_data
        
        class_weights = {'healthy': 1.0, 'chlorosis': 2.0, 'necrosis': 2.0}
        
        clf = XGBoostClassifier(n_estimators=10, class_weights=class_weights)
        clf.fit(X_train, y_train)
        
        # Should train without error
        predictions = clf.predict(X_val)
        assert len(predictions) == len(y_val)


class TestExtractFeaturesFromDirectory:
    """Tests for directory-based feature extraction."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset with class folders."""
        tmpdir = tempfile.mkdtemp()
        
        # Create class folders with images
        classes = ['healthy', 'chlorosis', 'necrosis']
        for cls in classes:
            class_dir = Path(tmpdir) / cls
            class_dir.mkdir()
            
            # Create dummy images
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(class_dir / f'image_{i}.png'), img)
        
        yield tmpdir
        
        # Cleanup
        shutil.rmtree(tmpdir)
    
    def test_extract_from_directory(self, temp_dataset):
        """Test extracting features from directory structure."""
        extractor = FeatureExtractor()
        
        features, names, labels = extract_features_from_directory(
            temp_dataset, extractor
        )
        
        # Should have at least 9 images (3 classes * 3 images)
        # May be more due to case-insensitive extension matching
        assert features.shape[0] >= 9  
        assert features.shape[1] == extractor.feature_dim
        assert len(names) == features.shape[0]
        assert len(labels) == features.shape[0]
        assert set(labels) == {'healthy', 'chlorosis', 'necrosis'}
    
    def test_extract_with_grayscale(self, temp_dataset):
        """Test directory extraction with grayscale option."""
        extractor = FeatureExtractor(use_grayscale=True)
        
        features, names, labels = extract_features_from_directory(
            temp_dataset, extractor
        )
        
        assert features.shape[0] >= 9


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete feature extraction -> training -> evaluation pipeline."""
        # Create dummy dataset
        np.random.seed(42)
        
        # Simulate extracted features
        n_train = 100
        n_val = 20
        n_features = 194
        
        X_train = np.random.randn(n_train, n_features)
        y_train = np.array(['healthy'] * 40 + ['chlorosis'] * 30 + ['necrosis'] * 30)
        
        X_val = np.random.randn(n_val, n_features)  
        y_val = np.array(['healthy'] * 8 + ['chlorosis'] * 6 + ['necrosis'] * 6)
        
        # Train model
        clf = XGBoostClassifier(n_estimators=20, max_depth=4)
        clf.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = clf.evaluate(X_val, y_val)
        
        print(f"\nIntegration test results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # Test should at least run without errors
        assert 'accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
