import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data

def load_models():
    """Load all trained models."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    models = {}
    
    model_files = [
        'random_forest.joblib',
        'gradient_boosting.joblib',
        'logistic_regression.joblib'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            model_name = model_file.split('.')[0]
            models[model_name] = joblib.load(model_path)
    
    return models

def compare_models():
    """Compare all trained models using ROC and Precision-Recall curves."""
    # Load models
    models = load_models()
    if not models:
        print("No trained models found. Please train models first.")
        return
    
    # Get test data
    X_train, X_test, y_train, y_test = preprocess_data(save=False)
    
    # Set up plots
    plt.figure(figsize=(12, 10))
    
    # ROC Curve subplot
    plt.subplot(2, 1, 1)
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall Curve subplot
    plt.subplot(2, 1, 2)
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    print(f"Model comparison plot saved to plots/model_comparison.png")
    
    return models

def recommend_best_model(models, X_test, y_test):
    """Recommend the best model based on multiple metrics."""
    if not models:
        print("No models to compare.")
        return None
    
    results = []
    
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate ROC AUC
        roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])
        
        # Calculate Average Precision
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        results.append({
            'model': model_name,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nModel Comparison Results:")
    print(results_df)
    
    # Determine best model based on ROC AUC and average precision
    best_by_roc = results_df.loc[results_df['roc_auc'].idxmax()]
    best_by_ap = results_df.loc[results_df['avg_precision'].idxmax()]
    
    print(f"\nBest model by ROC AUC: {best_by_roc['model']} (AUC = {best_by_roc['roc_auc']:.3f})")
    print(f"Best model by Average Precision: {best_by_ap['model']} (AP = {best_by_ap['avg_precision']:.3f})")
    
    # Overall recommendation (if same model wins both metrics)
    if best_by_roc['model'] == best_by_ap['model']:
        recommendation = best_by_roc['model']
    else:
        # If different models win different metrics, recommend the one with higher average rank
        results_df['roc_rank'] = results_df['roc_auc'].rank(ascending=False)
        results_df['ap_rank'] = results_df['avg_precision'].rank(ascending=False)
        results_df['avg_rank'] = (results_df['roc_rank'] + results_df['ap_rank']) / 2
        recommendation = results_df.loc[results_df['avg_rank'].idxmin()]['model']
    
    print(f"\nOverall recommended model: {recommendation}")
    return recommendation

if __name__ == "__main__":
    print("Comparing trained models...")
    models = compare_models()
    
    # Get test data for recommendation
    _, X_test, _, y_test = preprocess_data(save=False)
    
    # Recommend best model
    best_model = recommend_best_model(models, X_test, y_test)