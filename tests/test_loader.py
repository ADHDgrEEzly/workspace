import pytest
import pandas as pd
import os
from src.data_processing.loader import load_data, get_data_info


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })
    
    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_load_data_valid(sample_csv):
    """Test loading data with valid parameters."""
    X_train, X_test, y_train, y_test = load_data(
        filepath=sample_csv,
        target_column='target',
        test_size=0.4,
        random_state=42
    )
    
    # Check shapes
    assert len(X_train) == 3  # 60% of 5 = 3
    assert len(X_test) == 2   # 40% of 5 = 2
    assert len(y_train) == 3
    assert len(y_test) == 2
    
    # Check columns
    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns
    assert all(col in X_train.columns for col in ['feature1', 'feature2'])


def test_load_data_invalid_file():
    """Test loading data with invalid file path."""
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent.csv', 'target')


def test_load_data_missing_target(sample_csv):
    """Test loading data with missing target column."""
    with pytest.raises(ValueError, match="Target column 'wrong_target' not found in dataset"):
        load_data(sample_csv, 'wrong_target')


def test_load_data_invalid_test_size(sample_csv):
    """Test loading data with invalid test size."""
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        load_data(sample_csv, 'target', test_size=1.5)


def test_get_data_info(sample_csv):
    """Test getting data information."""
    data = pd.read_csv(sample_csv)
    info = get_data_info(data)
    
    assert info['n_samples'] == 5
    assert info['n_features'] == 3
    assert 'feature1' in info['numeric_features']
    assert 'feature2' in info['categorical_features']
    assert all(count == 0 for count in info['missing_values'].values())