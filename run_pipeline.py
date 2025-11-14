"""
Loan Default Prediction - End-to-End Pipeline

This script runs the complete loan default prediction pipeline:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Model explanation
5. Optional: Launch Streamlit app
"""

import os
import argparse
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_and_save
from src.train_model import train_models
from src.evaluate import evaluate_model
from src.explain import explain_model
import subprocess
import sys

def run_pipeline(skip_training=False, launch_app=False):
    """
    Run the complete loan default prediction pipeline
    
    Args:
        skip_training (bool): If True, skip model training (use saved model)
        launch_app (bool): If True, launch the Streamlit app after pipeline completes
    """
    print("=" * 80)
    print("LOAN DEFAULT PREDICTION PIPELINE")
    print("=" * 80)
    
    # Check if data exists
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'Loan_Default.csv')
    if not os.path.exists(data_path):
        print("Error: Dataset not found at:", data_path)
        print("Please place the Loan_Default.csv file in the data directory")
        return False
    
    # Step 1: Data Preprocessing
    print("\n[Step 1/4] Data preprocessing...")
    try:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_and_save()
        print("✓ Data preprocessing completed successfully")
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        return False
    
    # Step 2: Model Training (if not skipped)
    if not skip_training:
        print("\n[Step 2/4] Training models...")
        try:
            best_models, best_model_name = train_models()
            print(f"✓ Model training completed successfully")
            print(f"✓ Best model: {best_model_name}")
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return False
    else:
        print("\n[Step 2/4] Skipping model training (using saved model)")
        
    # Step 3: Model Evaluation
    print("\n[Step 3/4] Evaluating model...")
    try:
        metrics = evaluate_model()
        print("✓ Model evaluation completed successfully")
        print("Model performance metrics:")
        for metric_name, value in metrics.items():
            print(f"  - {metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        return False
    
    # Step 4: Model Explanation
    print("\n[Step 4/4] Generating model explanations...")
    try:
        explain_model()
        print("✓ Model explanations generated successfully")
    except Exception as e:
        print(f"Error in model explanation: {str(e)}")
        print("Continuing despite explanation error...")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    # Results location
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    print(f"\nResults saved to: {results_dir}")
    
    # Launch Streamlit app if requested
    if launch_app:
        print("\nLaunching Streamlit app...")
        try:
            app_path = os.path.join(os.path.dirname(__file__), 'src', 'app_streamlit.py')
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
            print("✓ Streamlit app launched")
        except Exception as e:
            print(f"Error launching Streamlit app: {str(e)}")
    else:
        print("\nTo launch the Streamlit app, run:")
        print(f"streamlit run {os.path.join('src', 'app_streamlit.py')}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Prediction Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training (use saved model)")
    parser.add_argument("--launch-app", action="store_true", help="Launch Streamlit app after pipeline completes")
    args = parser.parse_args()
    
    run_pipeline(skip_training=args.skip_training, launch_app=args.launch_app)