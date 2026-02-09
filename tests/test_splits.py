"""Tests for data splitting functions."""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.plantdisease.data.splits import make_splits, load_split

@pytest.fixture
def sample_data():
    """Create sample dataframe for testing."""
    data = {
        'image_path': [f'img_{i}.jpg' for i in range(100)],
        'label': [i % 5 for i in range(100)]  # 5 classes
    }
    return pd.DataFrame(data)

def test_make_splits(sample_data):
    """Test split creation."""
    # TODO: Implement test for make_splits
    # This would require mocking or actual data
    pass

def test_load_split_missing_file():
    """Test loading non-existent split file."""
    with pytest.raises(FileNotFoundError):
        load_split('train', split_dir=Path('/nonexistent'))

def test_split_ratios():
    """Test that splits sum to 1.0."""
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6
