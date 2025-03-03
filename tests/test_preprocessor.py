import pytest
import pandas as pd
import numpy as np
from src.data_processing.preprocessor import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'numeric1': [1.0, 2.0, np.nan, 4.0],
        'numeric2': [10.0, np.nan, 30.0, 40.0],
        'category1': ['A', 'B', np.nan, 'A'],
        'category2': ['X', 'Y', 'Z', np.nan]
    })


def test_handle_missing_values(sample_data):
    preprocessor = DataPreprocessor(numeric_strategy='mean', categorical_strategy='most_frequent')
    preprocessor._identify_features(sample_data)
    
    # Process the data
    processed_data = preprocessor.handle_missing_values(sample_data)
    
    # Check that there are no missing values
    assert processed_data.isnull().sum().sum() == 0
    
    # Check that numeric values are filled with mean
    assert processed_data.loc[2, 'numeric1'] == pytest.approx(sample_data['numeric1'].mean())
    assert processed_data.loc[1, 'numeric2'] == pytest.approx(sample_data['numeric2'].mean())
    
    # Check that categorical values are filled with mode
    assert processed_data.loc[2, 'category1'] == 'A'  # Most frequent value
    assert processed_data.loc[3, 'category2'] == 'X'  # First most frequent value


def test_encode_categorical_features(sample_data):
    preprocessor = DataPreprocessor()
    preprocessor._identify_features(sample_data)
    
    # First handle missing values
    clean_data = preprocessor.handle_missing_values(sample_data)
    
    # Encode categorical features
    encoded_data = preprocessor.encode_categorical_features(clean_data)
    
    # Check that categorical columns are encoded as numbers
    assert encoded_data['category1'].dtype in [np.int32, np.int64]
    assert encoded_data['category2'].dtype in [np.int32, np.int64]
    
    # Check that encoding is consistent
    assert encoded_data.loc[0, 'category1'] == encoded_data.loc[3, 'category1']
    assert len(encoded_data['category1'].unique()) == 2  # A and B
    assert len(encoded_data['category2'].unique()) == 3  # X, Y, and Z


def test_scale_features(sample_data):
    preprocessor = DataPreprocessor()
    preprocessor._identify_features(sample_data)
    
    # First handle missing values
    clean_data = preprocessor.handle_missing_values(sample_data)
    
    # Scale features
    scaled_data = preprocessor.scale_features(clean_data)
    
    # Check that numeric columns are scaled (mean ≈ 0, std ≈ 1)
    for col in ['numeric1', 'numeric2']:
        assert scaled_data[col].mean() == pytest.approx(0, abs=1e-10)
        assert scaled_data[col].std() == pytest.approx(1, abs=1e-10)
    
    # Check that categorical columns are unchanged
    assert scaled_data['category1'].equals(clean_data['category1'])
    assert scaled_data['category2'].equals(clean_data['category2'])


def test_fit_transform_and_transform(sample_data):
    preprocessor = DataPreprocessor()
    
    # Fit and transform training data
    transformed_train = preprocessor.fit_transform(sample_data)
    
    # Create test data with similar structure
    test_data = pd.DataFrame({
        'numeric1': [3.0, 5.0],
        'numeric2': [25.0, 35.0],
        'category1': ['B', 'A'],
        'category2': ['Y', 'X']
    })
    
    # Transform test data
    transformed_test = preprocessor.transform(test_data)
    
    # Check that all features are processed
    assert transformed_test.shape == test_data.shape
    assert not transformed_test.isnull().any().any()
    
    # Check that categorical encoding is consistent
    train_a_value = transformed_train.loc[0, 'category1']  # Value for 'A'
    test_a_value = transformed_test.loc[1, 'category1']  # Value for 'A'
    assert train_a_value == test_a_value