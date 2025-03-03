import pytest
import pandas as pd
import numpy as np
from src.model.logistic_model import LogisticModel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    # Create target variable with some relationship to features
    y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
    return X, y


def test_model_training(sample_data):
    """Test model training with valid data."""
    X, y = sample_data
    model = LogisticModel(random_state=42)
    
    # Train model
    model.train(X, y)
    
    # Check that model is fitted
    assert hasattr(model.model, 'coef_')
    assert hasattr(model.model, 'intercept_')
    
    # Check coefficient shape
    assert model.model.coef_.shape == (1, 2)  # Binary classification, 2 features


def test_model_predictions(sample_data):
    """Test model predictions."""
    X, y = sample_data
    model = LogisticModel(random_state=42)
    model.train(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})  # Binary classification
    
    # Test probability predictions
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_feature_importance(sample_data):
    """Test feature importance calculation."""
    X, y = sample_data
    model = LogisticModel(random_state=42)
    model.train(X, y)
    
    # Get feature importance
    importance = model.get_feature_importance(X.columns.tolist())
    
    # Check that we get importance for all features
    assert len(importance) == len(X.columns)
    assert all(importance.index == X.columns)
    
    # Check that importance values are non-negative
    assert np.all(importance >= 0)
    
    # Check that importance is sorted in descending order
    assert all(importance.iloc[i] >= importance.iloc[i + 1] 
              for i in range(len(importance) - 1))


def test_model_with_invalid_input(sample_data):
    """Test model behavior with invalid input."""
    X, y = sample_data
    model = LogisticModel()
    
    # Test training with mismatched shapes
    with pytest.raises(ValueError):
        model.train(X, y[:50])  # Mismatched lengths
    
    # Train model properly for prediction tests
    model.train(X, y)
    
    # Test prediction with wrong number of features
    X_invalid = pd.DataFrame({'feature1': [1, 2]})  # Missing feature2
    with pytest.raises(ValueError):
        model.predict(X_invalid)


def test_model_reproducibility():
    """Test model reproducibility with random_state."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    y = pd.Series(np.random.binomial(1, 0.5, 100))
    
    # Train two models with same random_state
    model1 = LogisticModel(random_state=42)
    model2 = LogisticModel(random_state=42)
    
    model1.train(X, y)
    model2.train(X, y)
    
    # Predictions should be identical
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    assert np.array_equal(pred1, pred2)
    assert np.array_equal(model1.model.coef_, model2.model.coef_)