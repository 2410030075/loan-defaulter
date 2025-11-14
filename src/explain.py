import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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

def get_raw_data_and_model():
    """Get raw data, processed data, and model without any SHAP-specific processing."""
    model = load_model()
    
    # Initialize preprocessor and load the data
    preprocessor = LoanDefaultPreprocessor()
    df = preprocessor.load_data()
    feature_columns, target_column = preprocessor.identify_features_and_target(df)
    
    # Get processed data
    X_train_processed, X_test, y_train, y_test = preprocess_data(save=False)
    
    # Get feature names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    else:
        # If not available, generate generic feature names based on the shape of X_test
        if hasattr(X_test, 'shape'):
            if len(X_test.shape) == 2:
                feature_names = [f"Feature_{i+1}" for i in range(X_test.shape[1])]
            else:
                feature_names = [f"Feature_{i+1}" for i in range(10)]
        else:
            feature_names = feature_columns
            
    return model, X_test, y_test, feature_names, preprocessor
    
def explain_model(sample_size=200):
    """Generate explanations for the model using SHAP."""
    model, X_test, y_test, feature_names, preprocessor = get_raw_data_and_model()
    
    # Convert data to numpy arrays to avoid indexing issues with DataFrames
    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = X_test
        
    if isinstance(y_test, pd.Series):
        y_test_np = y_test.values
    else:
        y_test_np = y_test
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    plots_dir = os.path.join(os.path.join(results_dir, 'plots'))
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use a sample for SHAP analysis to limit compute time
    if len(X_test) > sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
        y_sample = y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices]
    else:
        X_sample = X_test
        y_sample = y_test
    
    print(f"Using {len(X_sample)} samples for SHAP analysis")
    
    # Global feature importance
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Make sure feature_names is the right length
            if len(feature_names) != len(importances):
                print(f"Warning: Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")
                feature_names = [f"Feature_{i+1}" for i in range(len(importances))]
            
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            # Save feature importance to CSV
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
            print(f"Feature importance plot saved to {os.path.join(plots_dir, 'feature_importance.png')}")
            print(f"Feature importance CSV saved to {os.path.join(results_dir, 'feature_importance.csv')}")
            
    except Exception as e:
        print(f"Could not generate standard feature importance plot: {e}")
    
    # SHAP analysis
    try:
        print("\nGenerating SHAP explanations...")
        
        # Sample the numpy arrays
        if len(X_test_np) > sample_size:
            sample_indices = np.random.choice(len(X_test_np), sample_size, replace=False)
            X_sample = X_test_np[sample_indices]
            y_sample = y_test_np[sample_indices]
        else:
            X_sample = X_test_np
            y_sample = y_test_np
        
        print(f"Using {len(X_sample)} samples for SHAP analysis")
            
        # Select the appropriate explainer based on model type
        print("Creating SHAP explainer based on model type...")
        model_type = str(type(model))
        
        if 'RandomForest' in model_type or 'GradientBoosting' in model_type or 'XGB' in model_type:
            print("Using TreeExplainer for tree-based model")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, shap_values may be a list of length 2
            if isinstance(shap_values, list):
                print(f"Model returned {len(shap_values)} sets of SHAP values, using values for positive class")
                if len(shap_values) == 2:
                    # Use values for class 1 (default)
                    shap_values = shap_values[1]
        else:
            print("Using KernelExplainer (this may take some time)")
            # Fallback to KernelExplainer for any model
            if hasattr(model, 'predict_proba'):
                sample_data = shap.sample(X_sample, min(50, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict_proba, sample_data)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Class 1 (default)
            else:
                sample_data = shap.sample(X_sample, min(50, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, sample_data)
                shap_values = explainer.shap_values(X_sample)
        
        print("Creating SHAP visualization plots...")
        
        # Make sure feature_names matches the shape of the data
        if len(feature_names) != X_sample.shape[1]:
            print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data shape ({X_sample.shape[1]})")
            feature_names = [f"Feature_{i+1}" for i in range(X_sample.shape[1])]
        
        # Summary plot
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to {os.path.join(plots_dir, 'shap_summary.png')}")
        
        # Bar plot
        print("Generating SHAP bar plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP bar plot saved to {os.path.join(plots_dir, 'shap_bar.png')}")
        
        try:
            # Decision plots for a few examples
            print("Generating SHAP decision plots...")
            num_examples = min(5, len(X_sample))
            example_indices = np.random.choice(len(X_sample), num_examples, replace=False)
            
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            
            plt.figure(figsize=(20, num_examples*3))
            shap.decision_plot(expected_value,
                              shap_values[example_indices] if isinstance(shap_values, np.ndarray) 
                              else shap_values[example_indices],
                              X_sample.iloc[example_indices] if hasattr(X_sample, 'iloc') 
                              else X_sample[example_indices],
                              feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'shap_decision_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP decision plot saved to {os.path.join(plots_dir, 'shap_decision_plot.png')}")
            
            # Generate waterfall plots for the same examples
            print("Generating SHAP waterfall plots...")
            for i, idx in enumerate(example_indices):
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[idx] if isinstance(shap_values, np.ndarray) else shap_values[idx],
                    base_values=expected_value,
                    data=X_sample.iloc[idx] if hasattr(X_sample, 'iloc') else X_sample[idx],
                    feature_names=feature_names
                ), show=False)
                plt.title(f"Example {i+1} (True label: {y_sample.iloc[idx] if hasattr(y_sample, 'iloc') else y_sample[idx]})")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'shap_waterfall_example_{i+1}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"SHAP waterfall plot for example {i+1} saved to {os.path.join(plots_dir, f'shap_waterfall_example_{i+1}.png')}")
        except Exception as e:
            print(f"Warning: Could not generate decision and waterfall plots: {e}")
            
        # Save SHAP values to CSV
        print("Saving SHAP values to CSV...")
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df['actual_class'] = y_sample
        shap_df.to_csv(os.path.join(results_dir, 'shap_values.csv'), index=False)
        print(f"SHAP values saved to {os.path.join(results_dir, 'shap_values.csv')}")
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate model explanations')
    parser.add_argument('--sample', type=int, default=200, help='Number of samples to use for SHAP analysis')
    args = parser.parse_args()
    
    explain_model(sample_size=args.sample)
    print("Model explanations generated successfully!")