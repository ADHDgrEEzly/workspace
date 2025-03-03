"""
Example script demonstrating the usage of the data science project components.
"""

import pandas as pd
import numpy as np
from src.data_processing.loader import load_data, get_data_info
from src.data_processing.preprocessor import DataPreprocessor
from src.model.logistic_model import LogisticModel
from src.evaluation.metrics import generate_classification_report


def main():
    # Create sample dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.normal(45, 15, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'education_years': np.random.normal(16, 3, 1000),
        'occupation': np.random.choice(['professional', 'technical', 'admin'], 1000),
        'has_debt': np.random.choice(['yes', 'no'], 1000),
        'approved': None  # Target variable
    })
    
    # Create target variable with some relationship to features
    data['approved'] = (
        (data['age'] > 25) &
        (data['income'] > 40000) &
        (data['education_years'] >= 14)
    ).astype(int)
    
    # Save to CSV
    data.to_csv('sample_data.csv', index=False)
    
    # 1. Load and split data
    X_train, X_test, y_train, y_test = load_data(
        'sample_data.csv',
        target_column='approved',
        test_size=0.2,
        random_state=42
    )
    
    # Print data info
    print("\nData Information:")
    print("-" * 50)
    info = get_data_info(X_train)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 2. Preprocess data
    preprocessor = DataPreprocessor(
        numeric_strategy='mean',
        categorical_strategy='most_frequent',
        scaling=True
    )
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # 3. Train model
    model = LogisticModel(random_state=42)
    model.train(X_train_processed, y_train)
    
    # 4. Make predictions
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)
    
    # 5. Generate evaluation report
    print("\nModel Evaluation:")
    print("-" * 50)
    report = generate_classification_report(
        y_test,
        y_pred,
        y_prob,
        class_labels=['Rejected', 'Approved']
    )
    
    # Print metrics
    print("\nMetrics:")
    for metric, value in report['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(report['classification_report'])
    
    # 6. Feature importance
    importance = model.get_feature_importance(X_train_processed.columns)
    print("\nFeature Importance:")
    print("-" * 50)
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.4f}")


if __name__ == "__main__":
    main()