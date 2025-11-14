import os
import pandas as pd
import numpy as np

def analyze_dataset():
    # Define the expected dataset path
    data_path = 'data/Loan_Default.csv'
    alternative_path = 'data/loan_default.csv'

    # Check if file exists (case-sensitive check)
    file_exists = os.path.exists(data_path) or os.path.exists(alternative_path)

    # Set the actual path based on what exists
    actual_path = data_path if os.path.exists(data_path) else alternative_path if os.path.exists(alternative_path) else None

    if actual_path:
        print(f"✅ Dataset found at: {os.path.abspath(actual_path)}")
        
        # Get file size in MB
        file_size_bytes = os.path.getsize(actual_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Try to load the dataset with different encodings if necessary
        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(actual_path, encoding=encoding)
                print(f"Successfully loaded dataset with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                print(f"Failed to load with encoding: {encoding}")
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                break
        
        if df is not None:
            # Display basic information
            print(f"\nDataset shape: {df.shape[0]} rows and {df.shape[1]} columns")
            print("\nFirst 5 rows:")
            print(df.head())
            
            print("\nColumn list:")
            for i, col in enumerate(df.columns, 1):
                print(f"{i}. {col}")
                
            # Generate a summary report
            print("\n" + "="*50)
            print("DATASET SUMMARY")
            print("="*50)
            
            # Count data types
            type_counts = df.dtypes.value_counts()
            print(f"Data types in dataset:")
            for dtype, count in type_counts.items():
                print(f"- {dtype}: {count} columns")
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"\nMemory usage: {memory_usage:.2f} MB")
            
            # Missing values overview
            missing_values = df.isnull().sum()
            missing_cols = missing_values[missing_values > 0]
            if len(missing_cols) > 0:
                print(f"\nColumns with missing values: {len(missing_cols)}")
                for col, count in missing_cols.items():
                    print(f"- {col}: {count} missing values ({count/len(df)*100:.2f}%)")
            else:
                print("\nNo missing values found in the dataset.")
                
            # Data summary
            print("\n" + "="*50)
            print("COMPLETE SUMMARY REPORT")
            print("="*50)
            print(f"File path: {os.path.abspath(actual_path)}")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Number of rows: {df.shape[0]}")
            print(f"Number of columns: {df.shape[1]}")
            print(f"Column list: {', '.join(df.columns)}")
        else:
            print("❌ Failed to load the dataset with any encoding.")
    else:
        print("❌ Dataset not found.")
        print("To download the dataset manually from Kaggle:")
        print("1. Visit https://www.kaggle.com/datasets/yasserh/loan-default-dataset")
        print("2. Click 'Download' button")
        print("3. Save the file to the 'data' folder as 'Loan_Default.csv'")
        
        print("\nTo download using Kaggle CLI:")
        print("1. Install Kaggle CLI: pip install kaggle")
        print("2. Set up your Kaggle API token in ~/.kaggle/kaggle.json")
        print("3. Run the following command:")
        print("   kaggle datasets download -d yasserh/loan-default-dataset --path data")
        print("4. Unzip the downloaded file:")
        print("   Expand-Archive -Path data/loan-default-dataset.zip -DestinationPath data")
        print("5. Verify the file exists")

if __name__ == "__main__":
    analyze_dataset()