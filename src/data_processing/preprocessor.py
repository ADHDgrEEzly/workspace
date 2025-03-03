import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handle data preprocessing tasks including missing values and feature engineering."""
    
    def __init__(
        self,
        numeric_strategy: str = 'mean',
        categorical_strategy: str = 'most_frequent',
        scaling: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            numeric_strategy: Strategy for handling numeric missing values ('mean', 'median')
            categorical_strategy: Strategy for handling categorical missing values ('most_frequent', 'constant')
            scaling: Whether to apply feature scaling
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling = scaling
        
        # Initialize preprocessors
        self.numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Store column information
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
    
    def _identify_features(self, df: pd.DataFrame) -> None:
        """Identify numeric and categorical features."""
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the imputers or just transform
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Handle numeric features
        if self.numeric_features:
            if fit:
                numeric_data = self.numeric_imputer.fit_transform(df[self.numeric_features])
            else:
                numeric_data = self.numeric_imputer.transform(df[self.numeric_features])
            df[self.numeric_features] = numeric_data
        
        # Handle categorical features
        if self.categorical_features:
            if fit:
                categorical_data = self.categorical_imputer.fit_transform(df[self.categorical_features])
            else:
                categorical_data = self.categorical_imputer.transform(df[self.categorical_features])
            df[self.categorical_features] = categorical_data
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders or just transform
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        if fit:
            self.label_encoders = {}
        
        for column in self.categorical_features:
            if fit:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
            else:
                df[column] = self.label_encoders[column].transform(df[column])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler or just transform
            
        Returns:
            DataFrame with scaled features
        """
        if not self.scaling:
            return df
        
        df = df.copy()
        
        if self.numeric_features:
            if fit:
                scaled_data = self.scaler.fit_transform(df[self.numeric_features])
            else:
                scaled_data = self.scaler.transform(df[self.numeric_features])
            df[self.numeric_features] = scaled_data
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self._identify_features(df)
        df = self.handle_missing_values(df, fit=True)
        df = self.encode_categorical_features(df, fit=True)
        df = self.scale_features(df, fit=True)
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to new data using fitted parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.handle_missing_values(df, fit=False)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_features(df, fit=False)
        return df