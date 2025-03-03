from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: Optional list of class labels
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Add per-class metrics if labels are provided
    if class_labels is not None:
        for avg in ['macro', 'weighted']:
            metrics.update({
                f'precision_{avg}': precision_score(y_true, y_pred, average=avg),
                f'recall_{avg}': recall_score(y_true, y_pred, average=avg),
                f'f1_{avg}': f1_score(y_true, y_pred, average=avg)
            })
    
    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> str:
    """
    Generate confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: Optional list of class labels
        
    Returns:
        Base64 encoded string of the plot image
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode()


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Generate ROC curve plot for binary classification.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_labels: Optional list of class labels
        
    Returns:
        Tuple of (base64 encoded plot image, AUC score)
    """
    if y_prob.shape[1] != 2:
        raise ValueError("ROC curve is only available for binary classification")
    
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode(), roc_auc


def generate_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[List[str]] = None
) -> Dict:
    """
    Generate a comprehensive classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        class_labels: Optional list of class labels
        
    Returns:
        Dictionary containing the report
    """
    report = {
        'metrics': calculate_metrics(y_true, y_pred, class_labels),
        'confusion_matrix_plot': plot_confusion_matrix(y_true, y_pred, class_labels),
        'classification_report': classification_report(y_true, y_pred, target_names=class_labels)
    }
    
    # Add ROC curve for binary classification if probabilities are provided
    if y_prob is not None and y_prob.shape[1] == 2:
        roc_plot, auc_score = plot_roc_curve(y_true, y_prob, class_labels)
        report.update({
            'roc_curve_plot': roc_plot,
            'auc_score': auc_score
        })
    
    return report