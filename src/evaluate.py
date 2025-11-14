import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import preprocess_data, LoanDefaultPreprocessor

def load_model(model_name=None):
    """Load a trained model from the models directory.
    If no model_name is specified, it tries to find the best available model."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # If model_name is specified, try to load that model
    if model_name:
        return joblib.load(os.path.join(models_dir, f"{model_name}.joblib"))
    
    # Otherwise, look for available models in priority order
    available_models = ['gradient_boosting', 'random_forest', 'logistic_regression']
    
    for model_name in available_models:
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            print(f"Loading model: {model_name}")
            return joblib.load(model_path)
    
    # If no models found, raise an error
    raise FileNotFoundError("No trained models found in the models directory.")

def evaluate_model(model=None, X_test=None, y_test=None):
    if model is None or X_test is None or y_test is None:
        _, X_test, _, y_test = preprocess_data(save=False)
        model = load_model()
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1),
        'recall': recall_score(y_test, y_pred, pos_label=1),
        'f1_score': f1_score(y_test, y_pred, pos_label=1),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Generate PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='navy', linestyle='--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pr_curve.png'), dpi=300)
    plt.close()
    
    # Save detailed classification report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(classification_rep).transpose()
    df_report.to_csv(os.path.join(results_dir, 'classification_report.csv'))
    
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    print("-" * 30)
    print(f"Metrics saved to: {os.path.join(results_dir, 'metrics.json')}")
    print(f"Evaluation plots saved to: {plots_dir}")
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate_model()
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")