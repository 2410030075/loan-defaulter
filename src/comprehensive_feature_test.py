import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data

def compare_feature_engineering_across_models():
    """Compare model performance with and without feature engineering across multiple models."""
    # Configuration options to test
    configs = [
        {"name": "Base Model", "use_feature_engineering": False, "use_polynomial_features": False},
        {"name": "With Feature Engineering", "use_feature_engineering": True, "use_polynomial_features": False},
        {"name": "With Polynomial Features", "use_feature_engineering": True, "use_polynomial_features": True, "poly_degree": 2},
        {"name": "With Cubic Features", "use_feature_engineering": True, "use_polynomial_features": True, "poly_degree": 3}
    ]
    
    # Models to test
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
    }
    
    all_results = []
    
    for config in configs:
        print(f"\n{'-'*80}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'-'*80}")
        
        # Preprocess data according to configuration
        poly_degree = config.get("poly_degree", 2)  # Default to 2 if not specified
        X_train, X_test, y_train, y_test = preprocess_data(
            use_smote=True,
            use_feature_engineering=config["use_feature_engineering"],
            use_polynomial_features=config["use_polynomial_features"],
            poly_degree=poly_degree,
            save=False
        )
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name} model with {X_train.shape[1]} features...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'config': config['name'],
                'model': model_name,
                'num_features': X_train.shape[1],
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Print evaluation metrics
            print(f"{model_name} Performance Metrics:")
            print(f"- Number of features: {metrics['num_features']}")
            print(f"- Accuracy: {metrics['accuracy']:.4f}")
            print(f"- Precision: {metrics['precision']:.4f}")
            print(f"- Recall: {metrics['recall']:.4f}")
            print(f"- F1 Score: {metrics['f1_score']:.4f}")
            print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Print confusion matrix and classification report
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(cm)
            
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            all_results.append(metrics)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(all_results)
    
    # Create visualization of the results by model and configuration
    models_list = results_df['model'].unique()
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Plot metrics by model
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Reshape data for seaborn
        plot_data = results_df.pivot(index='config', columns='model', values=metric)
        
        # Plot heatmap
        sns.heatmap(plot_data, annot=True, fmt='.4f', cmap='viridis', vmin=0.5, vmax=1.0)
        plt.title(f'{metric.upper()} by Model and Configuration', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f'feature_engineering_{metric}_comparison.png'))
    
    # Plot feature count by configuration
    plt.figure(figsize=(10, 6))
    sns.barplot(x='config', y='num_features', hue='model', data=results_df)
    plt.title('Number of Features by Configuration', fontsize=14)
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_count_comparison.png'))
    
    print(f"\nComparison plots saved to plots/ directory")
    
    # Create a summary table by model
    summary_table = pd.pivot_table(
        results_df, 
        values=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
        index=['model', 'config'],
        aggfunc=np.mean
    )
    
    return summary_table

if __name__ == "__main__":
    results = compare_feature_engineering_across_models()
    print("\nFinal Results Summary:")
    print(results)