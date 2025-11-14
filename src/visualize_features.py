import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import LoanDefaultPreprocessor

def visualize_top_features():
    """Visualize correlations between top features and target variable."""
    print("Visualizing top features correlations...")
    
    # Create the preprocessor
    preprocessor = LoanDefaultPreprocessor(use_feature_engineering=True)
    
    # Load data
    df = preprocessor.load_data()
    
    # Apply feature engineering
    df_engineered = preprocessor.engineer_features(df)
    
    # Identify features and target
    _, target_column = preprocessor.identify_features_and_target(df_engineered)
    
    # Top features identified in feature importance analysis
    top_features = [
        'Interest_rate_spread',
        'Upfront_charges',
        'rate_of_interest',
        'property_value',
        'dtir1',
        'loan_to_income',  # Engineered feature
        'Credit_Score'
    ]
    
    # Ensure all features exist in the dataset
    available_features = [f for f in top_features if f in df_engineered.columns]
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the visualization style
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_features = available_features + [target_column]
    corr_df = df_engineered[corr_features].corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation between Top Features and Target Variable', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_features_correlation.png'))
    
    # Create pairplot for the top features and target
    plt.figure(figsize=(14, 10))
    pair_df = df_engineered[available_features + [target_column]].sample(n=min(5000, len(df_engineered)))
    g = sns.pairplot(
        pair_df, 
        hue=target_column, 
        palette='viridis',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'w'},
        diag_kws={'alpha': 0.6}
    )
    g.fig.suptitle('Pairwise Relationships between Top Features', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_features_pairplot.png'))
    
    # Create individual feature distributions by target class
    for feature in available_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_engineered,
            x=feature,
            hue=target_column,
            element="step",
            stat="density",
            common_norm=False,
            bins=30,
            alpha=0.6
        )
        plt.title(f'Distribution of {feature} by Loan Default Status', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{feature}_distribution.png'))
    
    print(f"Feature visualizations saved to plots/ directory")

if __name__ == "__main__":
    visualize_top_features()