from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from typing import Optional


class LogisticModel:
    """Wrapper class for logistic regression model with additional functionality."""
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        **kwargs
    ):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            **kwargs
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the logistic regression model.
        
        Args:
            X: Feature DataFrame
            y: Target variable Series
        """
        # TODO: Add input validation
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        # TODO: Add prediction validation
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probability estimates
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: list) -> pd.Series:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Series with feature importance scores
        """
        # TODO: Implement feature importance calculation
        # For logistic regression, can use coefficients
        return pd.Series(
            self.model.coef_[0],
            index=feature_names
        ).abs().sort_values(ascending=False)