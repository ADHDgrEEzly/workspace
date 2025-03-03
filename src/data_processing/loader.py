import pandas as pd
import os
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


def load_data(
    filepath: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and split into training and test sets.
    
    Args:
        filepath: Path to the CSV file
        target_column: Name of the target variable column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test splits
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If target_column is not in the dataset or if test_size is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found at: {filepath}")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    data = pd.read_csv(filepath)
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() < len(y) // 10 else None
    )
    
    return X_train, X_test, y_train, y_test


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'numeric_features': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return info