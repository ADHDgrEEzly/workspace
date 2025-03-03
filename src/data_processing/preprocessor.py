import pandas as pd
import numpy as np
from typing import Tuple


class DataPreprocessor:
    """Handle data preprocessing tasks including missing values and feature engineering."""
    
    def __init__(self):
        # TODO: Add any necessary preprocessing parameters
        pass
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # TODO: Implement missing value strategy
        # - Numerical: mean/median imputation
        # - Categorical: mode/constant imputation
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        # TODO: Implement categorical encoding
        # - One-hot encoding for nominal variables
        # - Label encoding for ordinal variables
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        # TODO: Implement feature scaling
        # from sklearn.preprocessing import StandardScaler
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df)
        df = self.scale_features(df)
        return df