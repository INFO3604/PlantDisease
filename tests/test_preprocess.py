"""Tests for preprocessing functions."""
import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from src.plantdisease.data.preprocess.resize_standardize import resize_and_standardize
from src.plantdisease.data.preprocess.denoise import denoise_median, denoise_bilateral

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

def test_resize_and_standardize(temp_image):
    """Test image resizing and standardization."""
    result = resize_and_standardize(str(temp_image), size=(224, 224), normalize=False)
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
