import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data

def compare_feature_engineering():
    """Compare model performance with and without feature engineering."""
    # Configuration options to test
    configs = [
        {"name": "Base Model", "use_feature_engineering": False, "use_polynomial_features": False},
        {"name": "With Feature Engineering", "use_feature_engineering": True, "use_polynomial_features": False},
        {"name": "With Polynomial Features", "use_feature_engineering": True, "use_polynomial_features": True}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'-'*50}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'-'*50}")
        
        # Preprocess data according to configuration
        X_train, X_test, y_train, y_test = preprocess_data(
            use_smote=True,
            use_feature_engineering=config["use_feature_engineering"],
            use_polynomial_features=config["use_polynomial_features"],
            save=False
        )
        
        # Train a Random Forest model (best performer from previous tests)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train the model
        print(f"Training Random Forest model with {X_train.shape[1]} features...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'name': config['name'],
            'num_features': X_train.shape[1],
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print evaluation metrics
        print(f"Model Performance Metrics:")
        print(f"- Number of features: {metrics['num_features']}")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- Precision: {metrics['precision']:.4f}")
        print(f"- Recall: {metrics['recall']:.4f}")
        print(f"- F1 Score: {metrics['f1_score']:.4f}")
        print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
        
        results.append(metrics)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Create a visualization of the results
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create a bar chart for each metric
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='name', y=metric, data=results_df, palette='viridis')
        plt.title(f'{metric.upper()}', fontsize=14)
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.ylim(0.5, 1.0)  # Set y-axis to start from 0.5 for better visualization
    
    plt.subplot(2, 3, 6)
    sns.barplot(x='name', y='num_features', data=results_df, palette='rocket')
    plt.title('Number of Features', fontsize=14)
    plt.xlabel('')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'feature_engineering_comparison.png'))
    print(f"\nComparison plot saved to plots/feature_engineering_comparison.png")
    
    return results_df

if __name__ == "__main__":
    results = compare_feature_engineering()
    print("\nFinal Results Comparison:")
    print(results[['name', 'num_features', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])