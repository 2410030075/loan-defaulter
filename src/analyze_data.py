import os
import pandas as pd
import numpy as np

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Loan_Default.csv')
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data()
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nTarget distribution:")
    if 'loan_default' in df.columns:
        print(df['loan_default'].value_counts())
    else:
        print("No 'loan_default' column found")