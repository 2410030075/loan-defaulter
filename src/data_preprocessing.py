import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class LoanDefaultPreprocessor:
    def __init__(self, use_smote=True, use_feature_engineering=True, use_polynomial_features=False, poly_degree=2, random_state=42):
        self.use_smote = use_smote
        self.use_feature_engineering = use_feature_engineering
        self.use_polynomial_features = use_polynomial_features
        self.poly_degree = poly_degree
        self.random_state = random_state
        self.preprocessor = None
        self.categorical_features = None
        self.numeric_features = None
        self.ordinal_features = None
        self.binary_features = None
        self.target_encoded_features = None
        self.engineered_features = []
        
    def load_data(self):
        """Load the Loan_Default.csv data file from the data directory."""
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Loan_Default.csv')
        df = pd.read_csv(data_path)
        return df
    
    def identify_features_and_target(self, df):
        """Identify features and target variable (Status)."""
        target_column = 'Status'  # Based on our analysis, 'Status' is the loan default indicator
        feature_columns = [col for col in df.columns if col != target_column]
        return feature_columns, target_column
    
    def identify_feature_types(self, X_train):
        """Identify different feature types for preprocessing."""
        # Numeric features (exclude ID which is an identifier)
        self.numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'ID' in self.numeric_features:
            self.numeric_features.remove('ID')
        if 'year' in self.numeric_features:
            self.numeric_features.remove('year')  # year is likely not a predictor
            
        # Categorical features
        self.categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Features with high cardinality for target encoding
        high_cardinality_threshold = 10
        self.target_encoded_features = [
            col for col in self.categorical_features 
            if X_train[col].nunique() > high_cardinality_threshold
        ]
        
        # Binary categorical features (for one-hot encoding)
        self.binary_features = [
            col for col in self.categorical_features 
            if col not in self.target_encoded_features and X_train[col].nunique() <= 2
        ]
        
        # Ordinal features (for ordinal encoding)
        self.ordinal_features = [
            col for col in self.categorical_features 
            if col not in self.target_encoded_features and col not in self.binary_features
        ]
        
        print(f"Numeric features: {len(self.numeric_features)}")
        print(f"Binary categorical features: {len(self.binary_features)}")
        print(f"Ordinal categorical features: {len(self.ordinal_features)}")
        print(f"Target encoded features: {len(self.target_encoded_features)}")
    
    def get_top_numeric_features(self, X_train, y_train, top_n=5):
        """Get the top n numeric features most correlated with the target."""
        # Calculate correlation with target for numeric features
        if not self.numeric_features:
            return []
            
        # Create a temporary DataFrame with numeric features and target
        df_temp = pd.concat([X_train[self.numeric_features], y_train], axis=1)
        target_name = y_train.name
        
        # Calculate correlation with target
        correlations = df_temp.corr()[target_name].abs().sort_values(ascending=False)
        
        # Get top features (excluding the target itself)
        top_features = correlations[correlations.index != target_name].head(top_n).index.tolist()
        
        print(f"Top {len(top_features)} numeric features for polynomial transformation: {top_features}")
        return top_features
    
    def create_preprocessing_pipeline(self, X_train, y_train):
        """Create a preprocessing pipeline for the data."""
        self.identify_feature_types(X_train)
        
        # Basic numeric pipeline with imputation and scaling
        numeric_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        
        # Add polynomial features if requested
        if self.use_polynomial_features and self.numeric_features:
            # Get top numeric features for polynomial transformation
            top_features = self.get_top_numeric_features(X_train, y_train)
            
            if top_features:
                # Standard pipeline for regular numeric features
                numeric_transformer = Pipeline(steps=numeric_steps)
                
                # Polynomial pipeline for top features
                poly_steps = numeric_steps.copy()
                poly_steps.append(('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)))
                poly_transformer = Pipeline(steps=poly_steps)
                
                # Split numeric features into top and other
                other_numeric = [col for col in self.numeric_features if col not in top_features]
                
                # Will add these transformers separately below
                numeric_transformers = [
                    ('num', numeric_transformer, other_numeric),
                    ('poly', poly_transformer, top_features)
                ]
            else:
                # No top features identified, use standard pipeline for all numeric features
                numeric_transformer = Pipeline(steps=numeric_steps)
                numeric_transformers = [('num', numeric_transformer, self.numeric_features)]
        else:
            # Standard pipeline for all numeric features
            numeric_transformer = Pipeline(steps=numeric_steps)
            numeric_transformers = [('num', numeric_transformer, self.numeric_features)] if self.numeric_features else []
        
        # Binary categorical pipeline with imputation and one-hot encoding
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Ordinal categorical pipeline with imputation and ordinal encoding
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Target encoding for high cardinality categorical features
        target_encoder_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encoder', TargetEncoder(handle_missing='return_nan', handle_unknown='return_nan'))
        ])
        
        transformers = []
        
        # Add transformers only if we have features of that type
        if self.use_polynomial_features and self.numeric_features:
            # numeric_transformers was defined earlier and contains both standard and polynomial transformers
            transformers.extend(numeric_transformers)
        elif self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        
        if self.binary_features:
            transformers.append(('bin', binary_transformer, self.binary_features))
            
        if self.ordinal_features:
            transformers.append(('ord', ordinal_transformer, self.ordinal_features))
            
        if self.target_encoded_features:
            # We need to fit the target encoder separately since it requires y
            te_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('target_encoder', TargetEncoder(handle_missing='return_nan', handle_unknown='return_nan'))
            ])
            for col in self.target_encoded_features:
                te_transformer.fit(X_train[[col]], y_train)
                transformers.append((f'te_{col}', te_transformer, [col]))
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop columns not specified (like ID)
        )
        
        return self.preprocessor
    
    def split_data(self, df, features, target, test_size=0.2):
        """Split data into train and test sets."""
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance in the training data."""
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
    
    def engineer_features(self, df):
        """Create engineered features based on domain knowledge."""
        df_copy = df.copy()
        self.engineered_features = []
        
        print("Applying feature engineering...")
        
        # 1. Debt-to-income ratio
        if 'dtir1' in df_copy.columns:
            print("- Using existing dtir1 (debt-to-income ratio) feature")
        elif all(col in df_copy.columns for col in ['total_debt', 'income']) and df_copy['income'].notnull().sum() > 0:
            print("- Creating debt-to-income ratio")
            df_copy['debt_to_income'] = df_copy['total_debt'] / df_copy['income'].replace(0, np.nan)
            self.engineered_features.append('debt_to_income')
        
        # 2. Loan-to-income ratio
        if all(col in df_copy.columns for col in ['loan_amount', 'income']) and df_copy['income'].notnull().sum() > 0:
            print("- Creating loan-to-income ratio")
            df_copy['loan_to_income'] = df_copy['loan_amount'] / df_copy['income'].replace(0, np.nan)
            self.engineered_features.append('loan_to_income')
        
        # 3. Total number of loans
        if all(col in df_copy.columns for col in ['existing_loans', 'previous_loans']):
            print("- Creating total loans count")
            df_copy['num_loans_total'] = df_copy['existing_loans'] + df_copy['previous_loans']
            self.engineered_features.append('num_loans_total')
        
        # 4. Credit utilization
        if all(col in df_copy.columns for col in ['outstanding_balance', 'credit_limit']):
            print("- Creating credit utilization")
            df_copy['credit_utilization'] = df_copy['outstanding_balance'] / df_copy['credit_limit'].replace(0, np.nan)
            self.engineered_features.append('credit_utilization')
        
        # 5. Create binned features for age 
        if 'age' in df_copy.columns:
            # Handle non-numeric age values using regex to extract age range midpoints
            if df_copy['age'].dtype == 'object':
                print("- Creating age_binned from categorical age")
                # Check if the age column contains age ranges like "25-34"
                if df_copy['age'].str.contains('-').any():
                    try:
                        # Extract the midpoint of the age range
                        age_ranges = df_copy['age'].str.extract(r'(\d+)-(\d+)')
                        df_copy['age_numeric'] = age_ranges.astype(float).mean(axis=1)
                        
                        # Create quartile bins from the numeric age
                        df_copy['age_binned'] = pd.qcut(df_copy['age_numeric'].dropna(), 
                                                      q=4, 
                                                      labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior'])
                        self.engineered_features.extend(['age_numeric', 'age_binned'])
                    except Exception as e:
                        print(f"  Warning: Could not create age_binned due to: {e}")
            else:
                try:
                    print("- Creating age_binned from numeric age")
                    df_copy['age_binned'] = pd.qcut(df_copy['age'].dropna(), 
                                                  q=4, 
                                                  labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior'])
                    self.engineered_features.append('age_binned')
                except Exception as e:
                    print(f"  Warning: Could not create age_binned due to: {e}")
        
        # 6. Create binned features for Credit_Score
        if 'Credit_Score' in df_copy.columns and df_copy['Credit_Score'].dtype in ['int64', 'float64']:
            print("- Creating credit_score_binned")
            df_copy['credit_score_binned'] = pd.cut(df_copy['Credit_Score'], 
                                                   bins=[0, 580, 670, 740, 800, float('inf')],
                                                   labels=['Very_Poor', 'Fair', 'Good', 'Very_Good', 'Excellent'])
            self.engineered_features.append('credit_score_binned')
        
        # 7. LTV ratio binned (if exists)
        if 'LTV' in df_copy.columns:
            print("- Creating LTV_binned")
            df_copy['LTV_binned'] = pd.cut(df_copy['LTV'], 
                                           bins=[0, 80, 90, 95, float('inf')],
                                           labels=['Low_Risk', 'Medium_Risk', 'High_Risk', 'Very_High_Risk'])
            self.engineered_features.append('LTV_binned')
            
        print(f"Feature engineering completed. Added {len(self.engineered_features)} new features.")
        return df_copy

    def preprocess_and_save(self, save=True):
        """Main method to preprocess data and save preprocessing pipeline."""
        # Load data
        df = self.load_data()
        
        # Apply feature engineering if requested
        if self.use_feature_engineering:
            df = self.engineer_features(df)
            
        # Identify features and target
        features, target = self.identify_features_and_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, features, target)
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline(X_train, y_train)
        
        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Transform test data
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Apply SMOTE to handle class imbalance if requested
        if self.use_smote:
            X_train_processed, y_train = self.apply_smote(X_train_processed, y_train)
            
        # Save preprocessor
        if save:
            preprocessing_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(preprocessing_dir, exist_ok=True)
            joblib.dump(self.preprocessor, os.path.join(preprocessing_dir, 'preprocessor.joblib'))
            
        return X_train_processed, X_test_processed, y_train, y_test

def preprocess_data(use_smote=True, use_feature_engineering=True, use_polynomial_features=False, poly_degree=2, save=True):
    """Convenience function to preprocess data.
    
    Args:
        use_smote (bool): Whether to apply SMOTE for class balancing.
        use_feature_engineering (bool): Whether to create engineered features.
        use_polynomial_features (bool): Whether to create polynomial interactions for top numeric features.
        poly_degree (int): Degree of polynomial features (2=quadratic, 3=cubic, etc).
        save (bool): Whether to save the preprocessor to disk.
        
    Returns:
        tuple: (X_train_processed, X_test_processed, y_train, y_test)
    """
    preprocessor = LoanDefaultPreprocessor(
        use_smote=use_smote,
        use_feature_engineering=use_feature_engineering,
        use_polynomial_features=use_polynomial_features,
        poly_degree=poly_degree
    )
    return preprocessor.preprocess_and_save(save)

if __name__ == "__main__":
    # Preprocess the data and print some information
    X_train_processed, X_test_processed, y_train, y_test = preprocess_data(
        use_smote=True, 
        use_feature_engineering=True,
        use_polynomial_features=False
    )
    
    print(f"\nOriginal class distribution in training data:")
    print(pd.Series(y_train).value_counts(normalize=True))
    
    print(f"\nPreprocessed data shape: X_train {X_train_processed.shape}, X_test {X_test_processed.shape}")