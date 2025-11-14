"""
Generate comprehensive EDA report for the Loan Default dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ydata_profiling import ProfileReport
from src.config import load_config

def generate_eda_report():
    """
    Generate a comprehensive EDA report using pandas-profiling
    
    Returns:
        str: Path to the generated HTML report
    """
    # Load config
    config = load_config()
    target_column = config['data']['target_column']
    
    # Set the path to the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'Loan_Default.csv')
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return None
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the dataset
    try:
        # Try different encodings if needed
        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1']
        df = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                print(f"Dataset loaded successfully using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("ERROR: Could not read dataset with any of the supported encodings")
            return None
        
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {str(e)}")
        return None
    
    print("Generating EDA report...")
    
    # Generate the profile report
    profile = ProfileReport(
        df, 
        title="Loan Default Dataset - EDA Report",
        explorative=True,
        html={'style': {'full_width': True}},
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True}
        }
    )
    
    # Save the report
    report_path = os.path.join(results_dir, 'eda_report.html')
    profile.to_file(report_path)
    
    # Save missing values to CSV
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_df = pd.concat([missing_values, missing_percent], axis=1, keys=['Total', 'Percent'])
    missing_df = missing_df[missing_df['Total'] > 0]
    missing_csv_path = os.path.join(results_dir, 'missing_values.csv')
    missing_df.to_csv(missing_csv_path)
    
    # Generate additional plots
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Target distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_column, data=df)
    plt.title(f'Target Variable ({target_column}) Distribution')
    plt.savefig(os.path.join(plots_dir, 'target_distribution.png'))
    
    # Correlation heatmap
    plt.figure(figsize=(20, 16))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    
    # Summary statistics
    summary_stats = df.describe().T
    summary_stats['missing'] = df.isnull().sum()
    summary_stats['missing_percent'] = df.isnull().sum() / len(df) * 100
    summary_stats_path = os.path.join(results_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_stats_path)
    
    print(f"EDA report generated successfully: {report_path}")
    print(f"Missing values report: {missing_csv_path}")
    print(f"Summary statistics: {summary_stats_path}")
    print(f"Additional plots saved to: {plots_dir}")
    
    return report_path

if __name__ == "__main__":
    generate_eda_report()