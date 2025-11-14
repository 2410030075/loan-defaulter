"""
Configuration module for loading and handling model configurations
"""

import os
import json

def load_config():
    """
    Load configuration from model_config.json
    
    Returns:
        dict: Configuration dictionary
    """
    # Find the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_path = os.path.join(project_dir, 'config', 'model_config.json')
    
    # Check if the file exists
    if not os.path.exists(config_path):
        # Create the config directory if it doesn't exist
        os.makedirs(os.path.join(project_dir, 'config'), exist_ok=True)
        
        # Create a default config file
        default_config = {
            "data": {
                "target_column": "Status",
                "test_size": 0.2,
                "random_state": 42
            },
            "preprocessing": {
                "numeric_imputer_strategy": "median",
                "categorical_imputer_strategy": "most_frequent",
                "scale_numeric_features": True
            },
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "params": {
                        "max_iter": 1000,
                        "random_state": 42
                    },
                    "grid_search": {
                        "C": [0.01, 0.1, 1, 10, 100],
                        "class_weight": [None, "balanced"]
                    }
                },
                "random_forest": {
                    "enabled": True,
                    "params": {
                        "random_state": 42
                    },
                    "grid_search": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "gradient_boosting": {
                    "enabled": True,
                    "params": {
                        "random_state": 42
                    },
                    "grid_search": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5]
                    }
                },
                "xgboost": {
                    "enabled": True,
                    "params": {
                        "random_state": 42
                    },
                    "grid_search": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5]
                    }
                },
                "lightgbm": {
                    "enabled": True,
                    "params": {
                        "random_state": 42
                    },
                    "grid_search": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5]
                    }
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "cv_folds": 5,
                "scoring": "f1"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
            
        print(f"Created default config file at: {config_path}")
    
    # Load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def update_config(config):
    """
    Update the configuration file
    
    Args:
        config (dict): Updated configuration dictionary
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_path = os.path.join(project_dir, 'config', 'model_config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated config file at: {config_path}")

if __name__ == "__main__":
    # Test the module
    config = load_config()
    print("Configuration loaded successfully:")
    print(f"Target column: {config['data']['target_column']}")
    print(f"Enabled models: {[model for model, settings in config['models'].items() if settings['enabled']]}")