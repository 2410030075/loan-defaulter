import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_preprocessing import preprocess_data

def train_models():
    X_train, X_test, y_train, y_test = preprocess_data()
    
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'xgboost': XGBClassifier(random_state=42),
        'lightgbm': LGBMClassifier(random_state=42)
    }
    
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced']
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'lightgbm': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    best_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        grid_search = GridSearchCV(
            model, param_grids[model_name],
            cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"{model_name} best parameters: {grid_search.best_params_}")
        print(f"{model_name} best score: {grid_search.best_score_}")
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model in best_models.items():
        joblib.dump(model, os.path.join(models_dir, f"{model_name}.joblib"))
    
    best_model_name = max(best_models.items(), key=lambda x: getattr(x[1], 'best_score_', 0))[0]
    best_model = best_models[best_model_name]
    
    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
    
    return best_models, best_model_name

if __name__ == "__main__":
    best_models, best_model_name = train_models()
    print(f"Best model: {best_model_name}")
    print("All models trained and saved successfully!")