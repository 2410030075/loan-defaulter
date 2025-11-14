# Machine Learning Approach

This document details the machine learning methodology used in the Loan Default Prediction project.

## Problem Definition

The problem is formulated as a binary classification task where the goal is to predict whether a loan applicant will default (1) or not (0) based on a set of features describing the applicant, loan, and property.

## Data Preprocessing

### Missing Value Treatment
- Numeric features: Imputed with median values
- Categorical features: Imputed with mode (most frequent value)

### Feature Engineering
1. **Standard Features**:
   - **Loan-to-income ratio** = loan_amount / income
   - **Age binning** = Categorizing age into meaningful groups
   - **Credit score binning** = Grouping credit scores into risk levels
   - **LTV binning** = Categorizing loan-to-value ratios
   - **Debt-to-income ratio** = Using existing dtir1 feature

2. **Advanced Features** (created but with limited impact for tree-based models):
   - **Polynomial feature interactions** for numeric features
   - **Log transformations** for skewed numeric variables

### Feature Encoding
- **Binary categorical features**: Standard binary encoding (0/1)
- **Ordinal categorical features**: Ordinal encoding preserving order
- **Nominal categorical features**: One-hot encoding for non-ordinal features

### Feature Scaling
- Standard scaling (mean=0, std=1) applied to all numeric features
- Crucial for distance-based models and models with regularization

### Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique) used to balance training data
- Adjusted class weights in model parameters

## Model Selection and Training

### Models Evaluated
1. **Logistic Regression**
   - Linear model with regularization
   - Good baseline and interpretable

2. **Random Forest**
   - Ensemble of decision trees
   - Good at handling non-linear relationships
   - Robust against overfitting

3. **Gradient Boosting**
   - Sequential ensemble of weak learners
   - Highest performance for this task
   - Implementation: XGBoost

### Hyperparameter Tuning
Grid search with cross-validation used to optimize:

1. **Logistic Regression**:
   - Regularization strength (C)
   - Penalty type (l1, l2)

2. **Random Forest**:
   - Number of trees
   - Maximum depth
   - Minimum samples per leaf
   - Feature selection criteria

3. **Gradient Boosting**:
   - Learning rate
   - Maximum tree depth
   - Subsample ratio
   - Regularization parameters

## Model Evaluation

### Performance Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted defaults that were actual defaults
- **Recall**: Proportion of actual defaults that were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve, measuring discrimination ability

### Cross-Validation
- 5-fold cross-validation to ensure model robustness
- Stratified sampling to maintain class distribution

## Model Explainability

### Feature Importance
- Global feature importance extracted from tree-based models
- Identifies most predictive features across all predictions

### SHAP (SHapley Additive exPlanations)
- Calculates contribution of each feature to individual predictions
- Provides local interpretability at the instance level
- Visualized through:
  - Summary plots
  - Waterfall diagrams
  - Decision plots

## Deployment

The model is deployed as part of a Streamlit web application that allows for:
- Interactive prediction for single loan applications
- Batch prediction for multiple applications
- Model explanation for predictions
- Customizable probability thresholds

## Future Improvements

1. **Additional Models**:
   - Neural networks
   - Support Vector Machines
   - Stacked ensemble models

2. **Feature Engineering**:
   - Time-based features if temporal data available
   - Geographic clustering
   - More domain-specific feature interactions

3. **Model Monitoring**:
   - Tracking prediction drift
   - Performance monitoring over time
   - Scheduled retraining pipeline