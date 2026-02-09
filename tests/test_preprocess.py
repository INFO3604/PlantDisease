"""Tests for preprocessing functions."""
import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from src.plantdisease.data.preprocess.resize_standardize import resize_image, preprocess_image
from src.plantdisease.data.preprocess.denoise import denoise_median, denoise_bilateral
from src.plantdisease.data.preprocess.grayscale import (
    to_grayscale,
    batch_to_grayscale,
    is_grayscale,
    grayscale_with_alpha,
)

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return img

@pytest.fixture
def temp_image(sample_image):
    """Create temporary image file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.jpg"
        cv2.imwrite(str(img_path), sample_image)
        yield img_path

def test_resize_image(sample_image):
    """Test image resizing."""
    result = resize_image(sample_image, target_size=(224, 224))
    assert result is not None
    assert result.shape[:2] == (224, 224)

def test_preprocess_image(temp_image):
    """Test image preprocessing (resize and standardize)."""
    result = preprocess_image(str(temp_image), target_size=(224, 224))
    assert result is not None
    assert result.shape[:2] == (224, 224)

def test_denoise_median(temp_image):
    """Test median denoising."""
    result = denoise_median(str(temp_image), kernel_size=5)
    assert result is not None
    assert result.shape == cv2.imread(str(temp_image)).shape

def test_denoise_bilateral(temp_image):
    """Test bilateral denoising."""
    result = denoise_bilateral(str(temp_image))
    assert result is not None
    assert result.shape == cv2.imread(str(temp_image)).shape

def test_denoise_with_output(temp_image):
    """Test denoising with output file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "denoised.jpg"
        denoise_median(str(temp_image), kernel_size=5, output_path=output_path)
        assert output_path.exists()


# ============ Grayscale Tests ============

class TestToGrayscale:
    """Tests for the to_grayscale function."""
    
    def test_grayscale_from_path(self, temp_image):
        """Test grayscale conversion from file path."""
        result = to_grayscale(temp_image)
        assert result is not None
        assert len(result.shape) == 2  # Single channel
        assert result.shape == (100, 100)
    
    def test_grayscale_from_array(self, sample_image):
        """Test grayscale conversion from numpy array."""
        result = to_grayscale(sample_image)
        assert result is not None
        assert len(result.shape) == 2
        assert result.shape == (100, 100)
    
    def test_grayscale_keep_channels(self, temp_image):
        """Test grayscale with keep_channels=True returns 3 channels."""
        result = to_grayscale(temp_image, keep_channels=True)
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape == (100, 100, 3)
        # All channels should be identical
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])
    
    def test_grayscale_opencv_method(self, temp_image):
        """Test OpenCV conversion method."""
        result = to_grayscale(temp_image, method="opencv")
        assert result is not None
        assert len(result.shape) == 2
    
    def test_grayscale_luminosity_method(self, temp_image):
        """Test luminosity conversion method."""
        result = to_grayscale(temp_image, method="luminosity")
        assert result is not None
        assert len(result.shape) == 2
    
    def test_grayscale_average_method(self, temp_image):
        """Test average conversion method."""
        result = to_grayscale(temp_image, method="average")
        assert result is not None
        assert len(result.shape) == 2
    
    def test_grayscale_lightness_method(self, temp_image):
        """Test lightness conversion method."""
        result = to_grayscale(temp_image, method="lightness")
        assert result is not None
        assert len(result.shape) == 2
    
    def test_grayscale_with_output(self, temp_image):
        """Test grayscale conversion with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gray.jpg"
            result = to_grayscale(temp_image, output_path=output_path)
            assert result is not None
            assert output_path.exists()
            # Verify saved file
            saved = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            assert saved is not None
            assert saved.shape == (100, 100)
    
    def test_grayscale_invalid_path(self):
        """Test with non-existent file."""
        result = to_grayscale("nonexistent.jpg")
        assert result is None
    
    def test_grayscale_already_gray(self, temp_image):
        """Test conversion of already grayscale image."""
        # First convert to grayscale
        gray = to_grayscale(temp_image)
        # Then convert again
        result = to_grayscale(gray)
        assert result is not None
        assert np.array_equal(gray, result)


class TestBatchToGrayscale:
    """Tests for batch grayscale conversion."""
    
    def test_batch_conversion(self):
        """Test batch conversion of multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create multiple test images
            paths = []
            for i in range(3):
                img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
                path = tmpdir / f"test_{i}.jpg"
                cv2.imwrite(str(path), img)
                paths.append(path)
            
            results = batch_to_grayscale(paths)
            assert len(results) == 3
            assert all(r is not None for r in results)
            assert all(len(r.shape) == 2 for r in results)
    
    def test_batch_with_output_dir(self):
        """Test batch conversion with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            
            # Create test images
            paths = []
            for i in range(2):
                img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
                path = input_dir / f"test_{i}.jpg"
                cv2.imwrite(str(path), img)
                paths.append(path)
            
            batch_to_grayscale(paths, output_dir=output_dir)
            
            # Check output files exist
            assert (output_dir / "test_0.jpg").exists()
            assert (output_dir / "test_1.jpg").exists()


class TestIsGrayscale:
    """Tests for is_grayscale detection."""
    
    def test_rgb_image_not_grayscale(self, temp_image):
        """Test that RGB image is not detected as grayscale."""
        assert is_grayscale(temp_image) == False
    
    def test_gray_image_is_grayscale(self, temp_image):
        """Test that grayscale image is detected correctly."""
        gray = to_grayscale(temp_image)
        assert is_grayscale(gray) == True
    
    def test_gray_array_is_grayscale(self):
        """Test 2D array is grayscale."""
        gray_array = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        assert is_grayscale(gray_array) == True
    
    def test_identical_channels_is_grayscale(self):
        """Test that 3-channel image with identical channels is grayscale."""
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        # Stack to create 3-channel grayscale
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert is_grayscale(gray_3ch) == True


class TestGrayscaleWithAlpha:
    """Tests for grayscale conversion preserving alpha channel."""
    
    def test_rgba_to_grayscale(self):
        """Test RGBA image conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create RGBA image
            rgba = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
            path = tmpdir / "test.png"
            cv2.imwrite(str(path), rgba)
            
            result = grayscale_with_alpha(path)
            assert result is not None
            assert result.shape == (50, 50, 4)  # Should keep 4 channels
    
    def test_rgb_to_grayscale_no_alpha(self, temp_image):
        """Test RGB image (no alpha) conversion."""
        result = grayscale_with_alpha(temp_image)
        assert result is not None
        # Should return 3-channel grayscale
        assert len(result.shape) == 3
