import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data

def train_model(model_name='random_forest', save=True):
    """
    Train a machine learning model for loan default prediction.
    
    Args:
        model_name: The model to train ('random_forest', 'gradient_boosting', or 'logistic_regression')
        save: Whether to save the trained model
    
    Returns:
        Trained model and evaluation metrics
    """
    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(use_smote=True, save=True)
    
    # Initialize model based on selection
    if model_name == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_name == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_name == 'logistic_regression':
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}. Choose from 'random_forest', 'gradient_boosting', or 'logistic_regression'")
    
    print(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print evaluation metrics
    print(f"\nModel Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_confusion_matrix.png'))
    
    # If random forest, plot feature importance
    if model_name == 'random_forest' or model_name == 'gradient_boosting':
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            feature_importances = model.feature_importances_
            
            # Plot feature importances (top 20)
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(feature_importances))
            indices = np.argsort(feature_importances)[-top_n:]
            plt.barh(range(top_n), feature_importances[indices])
            plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{model_name}_feature_importance.png'))
    
    # Save model
    if save:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(models_dir, f'{model_name}.joblib'))
        print(f"\nModel saved to models/{model_name}.joblib")
    
    return model, metrics

if __name__ == "__main__":
    # Train all models
    for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
        print(f"\n{'='*50}")
        print(f"Training {model_name} model")
        print(f"{'='*50}")
        model, metrics = train_model(model_name=model_name)