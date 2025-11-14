"""
MLflow tracking for model experiments

This script provides functions for tracking model experiments using MLflow.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_preprocessing import preprocess_and_save

def setup_mlflow():
    """
    Set up MLflow tracking server
    
    Returns:
        str: URI of the MLflow tracking server
    """
    # Set the tracking URI to a local directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow_dir = os.path.join(project_dir, 'mlruns')
    os.makedirs(mlflow_dir, exist_ok=True)
    
    tracking_uri = f"file://{mlflow_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    
    return tracking_uri

def log_model(model, model_name, params=None, X_test=None, y_test=None, feature_names=None):
    """
    Log a model to MLflow
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        params (dict): Parameters used for the model
        X_test: Test features for evaluation
        y_test: Test target for evaluation
        feature_names (list): List of feature names
    
    Returns:
        str: Run ID of the MLflow run
    """
    # Set up MLflow
    setup_mlflow()
    
    # Start a new run
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        if params:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        # Log metrics if test data is provided
        if X_test is not None and y_test is not None:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        # Log feature importance if available
        if feature_names is not None:
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                
                # Create a DataFrame of feature importances
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)
                
                # Save to CSV and log as artifact
                importance_path = 'feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                
                # Clean up
                os.remove(importance_path)
                
        # Return the run ID
        return run.info.run_id

def track_experiment(models_dict, experiment_name="Loan Default Prediction"):
    """
    Track multiple models in an MLflow experiment
    
    Args:
        models_dict (dict): Dictionary of model name -> model object
        experiment_name (str): Name of the MLflow experiment
    
    Returns:
        dict: Dictionary of model name -> run ID
    """
    # Set up MLflow
    setup_mlflow()
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_save(save=False)
    
    # Get feature names if possible
    try:
        # Get numerical feature names
        num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Get categorical feature names (with one-hot encoding)
        cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
        cat_feature_names = []
        for cat in cat_features:
            unique_values = X_train[cat].dropna().unique()
            for val in unique_values:
                cat_feature_names.append(f"{cat}_{val}")
        
        # Combine all feature names
        feature_names = num_features + cat_feature_names
    except:
        feature_names = None
    
    # Log each model
    run_ids = {}
    for model_name, model in models_dict.items():
        try:
            # Get model parameters
            params = model.get_params()
        except:
            params = {}
        
        # Log the model
        run_id = log_model(model, model_name, params, X_test, y_test, feature_names)
        run_ids[model_name] = run_id
    
    return run_ids

def register_best_model(run_id, model_name, experiment_name="Loan Default Prediction"):
    """
    Register the best model in the MLflow model registry
    
    Args:
        run_id (str): Run ID of the MLflow run
        model_name (str): Name of the model
        experiment_name (str): Name of the MLflow experiment
    
    Returns:
        str: Model version
    """
    # Set up MLflow
    setup_mlflow()
    
    # Register the model
    model_uri = f"runs:/{run_id}/{model_name}"
    registered_model = mlflow.register_model(model_uri, model_name)
    
    return registered_model.version

if __name__ == "__main__":
    # Example usage
    import joblib
    
    # Load models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    models = {}
    
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.joblib') and model_file != 'preprocessor.joblib':
            model_name = model_file.replace('.joblib', '')
            model_path = os.path.join(models_dir, model_file)
            models[model_name] = joblib.load(model_path)
    
    if models:
        print(f"Tracking {len(models)} models in MLflow...")
        run_ids = track_experiment(models)
        
        print("Models logged to MLflow:")
        for model_name, run_id in run_ids.items():
            print(f"- {model_name}: {run_id}")
            
        # Register the best model (assuming best_model.joblib exists)
        best_model_path = os.path.join(models_dir, 'best_model.joblib')
        if os.path.exists(best_model_path):
            best_model_name = 'best_model'
            best_model = joblib.load(best_model_path)
            
            best_run_id = log_model(best_model, best_model_name)
            version = register_best_model(best_run_id, best_model_name)
            
            print(f"\nBest model registered as version {version}")
    else:
        print("No models found in the models directory. Please train models first.")