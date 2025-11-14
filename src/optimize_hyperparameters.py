"""
Hyperparameter optimization using Optuna

This script performs hyperparameter optimization for the best model
using Optuna, which provides a more efficient search strategy than GridSearchCV.
"""

import os
import joblib
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from src.data_preprocessing import preprocess_and_save
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import load_config

def objective(trial, X_train, X_test, y_train, y_test, model_type):
    """
    Objective function for Optuna optimization
    
    Args:
        trial: Optuna trial
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        model_type: Type of model to optimize ('random_forest', 'xgboost', etc.)
        
    Returns:
        float: F1 score
    """
    if model_type == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
        
    elif model_type == 'gradient_boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }
        model = GradientBoostingClassifier(**params)
        
    elif model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
        model = XGBClassifier(**params)
        
    elif model_type == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': 42
        }
        model = LGBMClassifier(**params)
        
    elif model_type == 'logistic_regression':
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'max_iter': 1000,
            'random_state': 42
        }
        # Ensure compatible solver and penalty combinations
        if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
            params['solver'] = 'saga'
        elif params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            params['solver'] = 'saga'
        elif params['penalty'] is None and params['solver'] == 'liblinear':
            params['solver'] = 'lbfgs'
            
        model = LogisticRegression(**params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)
    
    return f1

def optimize_hyperparameters(model_type='xgboost', n_trials=100):
    """
    Optimize hyperparameters for the given model type
    
    Args:
        model_type (str): Type of model to optimize
        n_trials (int): Number of optimization trials
        
    Returns:
        tuple: Best params, best model, best score
    """
    print(f"Optimizing hyperparameters for {model_type} model...")
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_save(save=False)
    
    # Create the Optuna study
    study = optuna.create_study(direction='maximize', study_name=f"{model_type}_optimization")
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test, model_type),
        n_trials=n_trials
    )
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"Best F1 score: {best_value:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Create and train the best model
    if model_type == 'random_forest':
        best_model = RandomForestClassifier(**best_params)
    elif model_type == 'gradient_boosting':
        best_model = GradientBoostingClassifier(**best_params)
    elif model_type == 'xgboost':
        best_model = XGBClassifier(**best_params)
    elif model_type == 'lightgbm':
        best_model = LGBMClassifier(**best_params)
    elif model_type == 'logistic_regression':
        best_model = LogisticRegression(**best_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    best_model.fit(X_train, y_train)
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    
    # Plot optimization history
    plt.subplot(2, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    
    # Plot parameter importance
    plt.subplot(2, 2, 2)
    optuna.visualization.matplotlib.plot_param_importances(study)
    
    # Plot intermediate values
    plt.subplot(2, 2, 3)
    optuna.visualization.matplotlib.plot_intermediate_values(study)
    
    # Plot parallel coordinate
    plt.subplot(2, 2, 4)
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'plots')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"{model_type}_optimization.png"))
    
    # Save the best model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"{model_type}_optimized.joblib"
    joblib.dump(best_model, os.path.join(models_dir, model_filename))
    
    # Also save as best_model.joblib if this is the best model so far
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    
    if not os.path.exists(best_model_path) or best_value > get_best_score():
        joblib.dump(best_model, best_model_path)
        save_best_score(best_value, model_type)
    
    return best_params, best_model, best_value

def get_best_score():
    """
    Get the best score achieved so far
    
    Returns:
        float: Best score achieved so far, or 0 if no score is saved
    """
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    score_path = os.path.join(results_dir, 'best_score.txt')
    
    if os.path.exists(score_path):
        with open(score_path, 'r') as f:
            return float(f.readline().strip())
    
    return 0

def save_best_score(score, model_type):
    """
    Save the best score achieved so far
    
    Args:
        score (float): Best score achieved
        model_type (str): Type of model that achieved the best score
    """
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    score_path = os.path.join(results_dir, 'best_score.txt')
    model_path = os.path.join(results_dir, 'best_model.txt')
    
    with open(score_path, 'w') as f:
        f.write(f"{score}")
    
    with open(model_path, 'w') as f:
        f.write(model_type)

if __name__ == "__main__":
    config = load_config()
    
    # Get enabled models
    enabled_models = [model for model, settings in config['models'].items() if settings['enabled']]
    
    # Optimize each enabled model
    for model in enabled_models:
        optimize_hyperparameters(model_type=model, n_trials=50)
    
    print("Hyperparameter optimization completed for all enabled models.")