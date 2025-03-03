import pandas as pd
from typing import Tuple, Optional


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
    """
    # TODO: Implement data validation and error handling
    data = pd.read_csv(filepath)
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # TODO: Implement train-test split
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(...)
    
    return X_train, X_test, y_train, y_test