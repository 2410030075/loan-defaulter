import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data, LoanDefaultPreprocessor

def analyze_feature_importance():
    """Analyze the importance of original and engineered features."""
    print("Analyzing feature importance with engineered features...")
    
    # Create the preprocessor
    preprocessor = LoanDefaultPreprocessor(
        use_smote=True,
        use_feature_engineering=True, 
        use_polynomial_features=False
    )
    
    # Load data
    df = preprocessor.load_data()
    
    # Apply feature engineering
    df_engineered = preprocessor.engineer_features(df)
    
    # Identify features and target
    features, target = preprocessor.identify_features_and_target(df_engineered)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_engineered, features, target)
    
    # Create preprocessing pipeline
    preprocessor.create_preprocessing_pipeline(X_train, y_train)
    
    # Fit and transform training data
    X_train_processed = preprocessor.preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.preprocessor.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    X_train_resampled, y_train_resampled = preprocessor.apply_smote(X_train_processed, y_train)
    
    # Train a Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    # Get feature names (original + engineered)
    feature_names = []
    
    # Extract feature names from preprocessor
    for name, transformer, column in preprocessor.preprocessor.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                # For transformers that generate new feature names
                for feat_name in transformer.get_feature_names_out():
                    feature_names.append(feat_name)
            else:
                # For simple transformers like StandardScaler
                for col in column:
                    feature_names.append(col)
    
    # Get the feature importance scores directly from the Random Forest model
    if len(feature_names) == X_train_processed.shape[1]:
        # Create dataframe of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 20 Features by Random Forest Importance:")
        print(feature_importance_df.head(20))
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title('Feature Importance (Random Forest)', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'feature_importance_analysis.png'))
    else:
        print(f"Warning: Feature names count ({len(feature_names)}) doesn't match feature count in data ({X_train_processed.shape[1]})")
    
    # Calculate permutation importance (more robust method)
    try:
        print("\nCalculating permutation importance (this may take some time)...")
        perm_importance = permutation_importance(
            model, X_test_processed, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create dataframe of permutation importances
        if len(feature_names) == X_train_processed.shape[1]:
            perm_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 20 Features by Permutation Importance:")
            print(perm_importance_df.head(20))
            
            # Plot permutation importance
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(20))
            plt.title('Permutation Feature Importance', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, 'permutation_importance_analysis.png'))
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
    
    # Analyze how engineered features rank compared to original features
    if preprocessor.engineered_features:
        print("\nAnalyzing engineered features importance:")
        for feature in preprocessor.engineered_features:
            if feature in feature_importance_df['Feature'].values:
                rank = feature_importance_df[feature_importance_df['Feature'] == feature].index[0] + 1
                importance = feature_importance_df[feature_importance_df['Feature'] == feature]['Importance'].values[0]
                print(f"- {feature}: Rank {rank} (Importance: {importance:.6f})")
            else:
                # Check if the feature has been encoded
                matches = feature_importance_df[feature_importance_df['Feature'].str.contains(feature, regex=False)]
                if not matches.empty:
                    print(f"- {feature}: Found as {len(matches)} encoded features:")
                    for i, row in matches.iterrows():
                        print(f"  - {row['Feature']}: Rank {i+1} (Importance: {row['Importance']:.6f})")
                else:
                    print(f"- {feature}: Not found in feature importance ranking")

if __name__ == "__main__":
    analyze_feature_importance()