# Feature Engineering for Loan Default Prediction

This module adds comprehensive feature engineering capabilities to the loan default prediction project.

## Key Features

1. **Standard Feature Engineering**
   - Debt-to-income ratio (existing feature: dtir1)
   - Loan-to-income ratio
   - Age binning (numeric age to age categories)
   - Credit score binning (numeric scores to risk categories)
   - LTV binning (loan-to-value ratio to risk categories)

2. **Advanced Feature Engineering**
   - Polynomial features (quadratic and cubic)
   - Feature interaction terms
   - Top feature selection for transformations

## Usage Examples

### Basic Usage with Standard Features

```python
from src.data_preprocessing import preprocess_data

# Preprocess data with standard feature engineering
X_train, X_test, y_train, y_test = preprocess_data(
    use_smote=True,
    use_feature_engineering=True,
    use_polynomial_features=False
)
```

### Advanced Usage with Polynomial Features

```python
from src.data_preprocessing import preprocess_data

# Preprocess data with polynomial features (degree 2)
X_train, X_test, y_train, y_test = preprocess_data(
    use_smote=True,
    use_feature_engineering=True,
    use_polynomial_features=True,
    poly_degree=2
)
```

## Implementation Details

The feature engineering is implemented in the `LoanDefaultPreprocessor` class with the following methods:

- `engineer_features(df)`: Creates engineered features based on domain knowledge
- `create_preprocessing_pipeline(X_train, y_train)`: Sets up a preprocessing pipeline that includes feature engineering
- `get_top_numeric_features(X_train, y_train)`: Identifies top features for polynomial transformation

## Performance Impact

Our comprehensive testing has shown:

1. **Random Forest and Gradient Boosting**:
   - Minimal improvement with feature engineering (already near-perfect performance)
   - No significant benefit from polynomial features

2. **Logistic Regression**:
   - Standard Feature Engineering: +1.3% F1 score, +0.8% ROC AUC
   - Polynomial Features (Degree 2): +4.3% F1 score, +2.7% ROC AUC
   - Cubic Features (Degree 3): +3.0% F1 score, +2.1% ROC AUC

## Feature Importance

The top 5 most important features for predicting loan default are:

1. Interest_rate_spread (0.3112)
2. Upfront_charges (0.2632)
3. rate_of_interest (0.1959)
4. credit_type (0.0539)
5. property_value (0.0386)

Our engineered feature "loan_to_income" ranked 11th in importance (0.0041).

## Testing

Comprehensive feature engineering tests can be run with:

```bash
python -m src.comprehensive_feature_test
```

For feature importance analysis:

```bash
python -m src.feature_importance_analysis
```

For feature visualizations:

```bash
python -m src.visualize_features
```