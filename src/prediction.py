import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import LoanDefaultPreprocessor

class LoanDefaultPredictor:
    def __init__(self, model_name='random_forest'):
        """Initialize the predictor with the specified model."""
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{model_name}.joblib')
        self.preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'preprocessor.joblib')
        
        # Load model and preprocessor
        self.load_model()
    
    def load_model(self):
        """Load the model and preprocessor from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        print(f"Loaded {self.model_name} model and preprocessor successfully.")
    
    def preprocess_input(self, data):
        """Preprocess input data using the saved preprocessor."""
        # Handle single row as dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame or a dictionary.")
        
        # Apply preprocessor
        processed_data = self.preprocessor.transform(data)
        return processed_data
    
    def predict(self, data):
        """Make binary predictions (0 or 1)."""
        processed_data = self.preprocess_input(data)
        return self.model.predict(processed_data)
    
    def predict_proba(self, data):
        """Make probability predictions (returns probability of default)."""
        processed_data = self.preprocess_input(data)
        probabilities = self.model.predict_proba(processed_data)
        return probabilities[:, 1]  # Return probability of class 1 (default)
    
    def explain_prediction(self, data):
        """Provide a simple explanation of the prediction based on input features."""
        # This is a simplified explanation - for more complex explanations consider using
        # explainable AI libraries like SHAP or LIME
        
        prediction = self.predict(data)[0]
        probability = self.predict_proba(data)[0]
        
        result = {
            'prediction': int(prediction),
            'default_probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'confidence': float(abs(probability - 0.5) * 2)  # Scale from 0 to 1
        }
        
        # Add explanation based on key risk factors
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
            # This is a placeholder - in a real scenario, we'd have feature names
            result['key_factors'] = [f"Feature {i}" for i in np.argsort(feature_importances)[-3:]]
        
        return result

def predict_loan_default(loan_data, model_name='random_forest'):
    """Convenience function to predict loan default probability."""
    predictor = LoanDefaultPredictor(model_name=model_name)
    return predictor.predict_proba(loan_data)[0]

if __name__ == "__main__":
    # Let's first get a sample from the dataset to understand the required columns
    loan_processor = LoanDefaultPreprocessor(use_smote=False)
    df = loan_processor.load_data()
    sample_row = df.iloc[0].to_dict()
    
    # Create a sample loan based on the actual dataset structure
    sample_loan = {
        'ID': 24890,
        'year': 2019,
        'loan_limit': 'cf',
        'Gender': 'Male',
        'approv_in_adv': 'nopre',
        'loan_type': 'type1',
        'loan_purpose': 'p1',
        'Credit_Worthiness': 'l1',
        'open_credit': 'nopc',
        'business_or_commercial': 'nob/c',
        'loan_amount': 250000,
        'rate_of_interest': 4.25,
        'Interest_rate_spread': 0.5,
        'Upfront_charges': 500.0,
        'term': 360.0,
        'Neg_ammortization': 'not_neg',
        'interest_only': 'not_int',
        'lump_sum_payment': 'not_lpsm',
        'property_value': 320000.0,
        'construction_type': 'sb',
        'occupancy_type': 'pr',
        'Secured_by': 'home',
        'total_units': '1U',
        'income': 6500,
        'credit_type': 'EXP',
        'Credit_Score': 680,
        'co-applicant_credit_type': 'CIB',
        'age': '35-44',
        'submission_of_application': 'to_inst',
        'LTV': 82.5,
        'Region': 'North',
        'Security_Type': 'direct',
        'dtir1': 38.0,
        # Status (target) is not included as it's what we're predicting
    }
    
    # Create predictor
    predictor = LoanDefaultPredictor(model_name='random_forest')
    
    # Get prediction and explanation
    explanation = predictor.explain_prediction(sample_loan)
    
    # Print results
    print("\nLoan Default Prediction Results:")
    print(f"Prediction: {'Default' if explanation['prediction'] == 1 else 'No Default'}")
    print(f"Default Probability: {explanation['default_probability']:.2%}")
    print(f"Risk Level: {explanation['risk_level']}")
    print(f"Confidence: {explanation['confidence']:.2%}")
    
    if 'key_factors' in explanation:
        print("Key Risk Factors:")
        for factor in explanation['key_factors']:
            print(f"- {factor}")
            
    # Try with different models
    print("\nComparing model predictions:")
    for model in ['random_forest', 'gradient_boosting', 'logistic_regression']:
        prob = predict_loan_default(sample_loan, model_name=model)
        print(f"{model}: {prob:.2%} chance of default")