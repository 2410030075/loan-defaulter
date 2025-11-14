import os
import pandas as pd

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Loan_Default.csv')
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data()
    
    print(f"All columns: {list(df.columns)}")
    
    # The 'Status' column might be the target variable
    if 'Status' in df.columns:
        print("\nStatus column distribution:")
        print(df['Status'].value_counts())
        print(f"\nStatus column dtype: {df['Status'].dtype}")
        print(f"\nUnique values in Status: {df['Status'].unique()}")