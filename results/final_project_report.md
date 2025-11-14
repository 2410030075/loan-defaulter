# Loan Default Prediction Project - Final Report

## Project Overview

This project focuses on predicting loan defaults using machine learning techniques with an emphasis on advanced feature engineering. We explored multiple approaches to feature engineering and evaluated their impact on different classification models.

## Dataset

The dataset contains loan information with features such as:
- Loan amount, interest rate, and term
- Borrower information (income, credit score)
- Property value and LTV ratio
- Various categorical variables (loan purpose, gender, etc.)

## Feature Engineering Implementation

We implemented the following feature engineering techniques:

### Custom Features
1. **Loan-to-income ratio**: Relationship between loan amount and borrower income
2. **Age binning**: Categorized age into interpretable groups
3. **Credit score binning**: Grouped credit scores into risk categories
4. **LTV binning**: Categorized loan-to-value ratios by risk level
5. **Debt-to-income ratio**: Used existing feature (dtir1)

### Feature Transformations
1. **Polynomial feature interactions**: Created quadratic and cubic combinations of top numeric features
2. **Standard scaling**: Normalized numeric features
3. **Encoding**: Applied appropriate encoding to categorical variables

## Performance Results

### Model Accuracy Comparison
| Model | Base | Feature Eng. | Polynomial | Cubic |
|-------|------|-------------|------------|-------|
| RandomForest | 0.9999 | 1.0000 | 0.9999 | 0.9998 |
| GradientBoosting | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| LogisticRegression | 0.6984 | 0.7035 | 0.7145 | 0.7057 |

### F1-Score Comparison
| Model | Base | Feature Eng. | Polynomial | Cubic |
|-------|------|-------------|------------|-------|
| RandomForest | 0.9999 | 0.9999 | 0.9997 | 0.9996 |
| GradientBoosting | 0.9998 | 0.9998 | 0.9999 | 0.9998 |
| LogisticRegression | 0.5299 | 0.5369 | 0.5528 | 0.5459 |

### ROC AUC Comparison
| Model | Base | Feature Eng. | Polynomial | Cubic |
|-------|------|-------------|------------|-------|
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LogisticRegression | 0.7648 | 0.7710 | 0.7854 | 0.7807 |

## Feature Importance Analysis

### Top 5 Most Important Features
1. **Interest_rate_spread**: 0.3112
2. **Upfront_charges**: 0.2632
3. **rate_of_interest**: 0.1959
4. **credit_type**: 0.0539
5. **property_value**: 0.0386

### Engineered Features Importance
- **loan_to_income**: Ranked 11th with importance of 0.0041
- **age_numeric**: Ranked 12th with importance of 0.0004
- **age_binned**, **credit_score_binned**, **LTV_binned**: Not found in top features

## Key Insights

1. **Tree-based models** (Random Forest and Gradient Boosting) perform exceptionally well on this dataset even without extensive feature engineering, achieving near-perfect scores across all metrics.

2. **Standard feature engineering** provides modest improvements, especially for Logistic Regression models:
   - F1 score improved from 0.5299 to 0.5369
   - ROC AUC improved from 0.7648 to 0.7710

3. **Polynomial feature engineering** provides the best boost for linear models:
   - F1 score improved from 0.5299 to 0.5528 (+4.3%)
   - ROC AUC improved from 0.7648 to 0.7854 (+2.7%)

4. **Cubic features** provide minimal additional benefit over quadratic features despite adding significant complexity (96 vs 61 features).

5. **Interest rate related features** (Interest_rate_spread, Upfront_charges, rate_of_interest) are the strongest predictors of loan default, regardless of the feature engineering approach.

## Visualizations

We created multiple visualizations to analyze the relationship between features and loan default:

1. **Feature correlation heatmap**: Shows correlation between top features and target variable
2. **Feature distributions by class**: Shows how each feature's distribution differs between default/non-default loans
3. **Feature importance plots**: Shows the relative importance of features in predicting loan default
4. **Model comparison plots**: Compares model performance across different feature engineering approaches

## Recommendations

1. **Use Random Forest** as the primary model for this loan default prediction task
2. **Apply standard feature engineering** for most production use cases, as it improves performance without excessive complexity
3. **Consider polynomial features** only for linear models like Logistic Regression
4. **Focus on interest rate features** when building simpler models or when interpretability is important
5. **Implement a hybrid approach** for production:
   - Train a robust tree-based model (Random Forest/Gradient Boosting)
   - Use feature engineering for improved interpretability
   - Provide feature importance information for decision-making

## Future Work

1. **Cross-validation**: Implement k-fold cross-validation to ensure results generalize well
2. **Hyperparameter tuning**: Optimize model parameters with the selected feature engineering approach
3. **Feature selection**: Investigate if a subset of features can achieve similar performance
4. **Explainable AI**: Implement SHAP values for better model interpretability
5. **Class imbalance exploration**: Test different class balancing techniques beyond SMOTE