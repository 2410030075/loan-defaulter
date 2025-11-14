import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model(model_name="gradient_boosting"):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # Fall back to other available models if specified one doesn't exist
    available_models = ['gradient_boosting', 'random_forest', 'logistic_regression']
    for name in available_models:
        model_path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(model_path):
            return joblib.load(model_path)
    
    raise FileNotFoundError("No trained models found in the models directory.")

# Load the preprocessor
@st.cache_resource
def load_preprocessor():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    return joblib.load(os.path.join(models_dir, "preprocessor.joblib"))

# Load schema
@st.cache_data
def load_schema():
    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'schema.json')
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            return json.load(f)
    raise FileNotFoundError("schema.json not found in the models directory.")
# Initialize SHAP explainer
@st.cache_resource
def get_explainer(_model):
    if hasattr(_model, 'estimators_') or 'RandomForest' in str(type(_model)) or 'GradientBoosting' in str(type(_model)):
        return shap.TreeExplainer(_model)
    else:
        return None

# Process input data for prediction
def preprocess_data(input_data, preprocessor):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)
    
    return preprocessor.transform(df)

# Get SHAP values for a prediction
def get_shap_values(model, explainer, input_processed, feature_names):
    if explainer is None:
        return None
    
    shap_values = explainer.shap_values(input_processed)
    
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # For binary classification, use positive class
    
    # Map SHAP values to feature names
    if len(feature_names) != shap_values.shape[1]:
        feature_names = [f"Feature {i+1}" for i in range(shap_values.shape[1])]
    
    return shap_values, feature_names

# Get top contributing features
def get_top_features(shap_values, feature_names, row_index=0, top_n=5):
    if shap_values is None:
        return None
    
    row_shap_values = shap_values[row_index]
    feature_importance = [(feature_names[i], abs(row_shap_values[i]), row_shap_values[i]) for i in range(len(feature_names))]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return feature_importance[:top_n]

# Main function
def main():
    try:
        model = load_model()
        preprocessor = load_preprocessor()
        schema = load_schema()
        explainer = get_explainer(model)
        
        if hasattr(preprocessor, 'feature_names_in_'):
            feature_names = preprocessor.feature_names_in_
        else:
            # Fallback to feature names from schema
            feature_names = [f['name'] for f in schema['numeric_features']] + [f['name'] for f in schema['categorical_features']]
        
        st.title("Loan Default Prediction App")
        
        tabs = st.tabs(["Instructions", "Single Prediction", "Batch Prediction"])
        
        with tabs[0]:
            st.header("How to Use This App")
            st.markdown("""
            This application predicts whether a loan will default based on various borrower and loan characteristics.
            
            ### Single Prediction Mode
            1. Enter the loan and borrower information in the form
            2. Click the 'Predict' button to see the prediction result
            3. The app will show:
               - The predicted probability of default
               - The prediction class (Default/No Default)
               - Top 5 features that influenced the prediction
            
            ### Batch Prediction Mode
            1. Upload a CSV file with multiple loan applications
            2. The CSV must contain the same fields as the prediction form
            3. Click 'Run Batch Prediction' to process all records
            4. Download the results as a CSV file
            
            ### Notes
            - The model has been trained on historical loan data
            - All data entered is processed locally and not stored
            """)
        
        with tabs[1]:
            st.header("Enter Loan Application Details")
            
            col1, col2 = st.columns(2)
            
            input_data = {}
            
            with col1:
                st.subheader("Numeric Fields")
                for feature in schema['numeric_features']:
                    input_data[feature['name']] = st.number_input(
                        f"{feature['name']}",
                        min_value=feature.get('min', 0.0),
                        max_value=feature.get('max', 1e6),
                        value=feature.get('default', 0.0),
                        step=1.0 if 'term' in feature['name'].lower() or 'age' in feature['name'].lower() else 0.01,
                        format="%d" if 'term' in feature['name'].lower() or 'age' in feature['name'].lower() else "%.2f"
                    )
            
            with col2:
                st.subheader("Categorical Fields")
                for feature in schema['categorical_features']:
                    input_data[feature['name']] = st.selectbox(
                        f"{feature['name']}",
                        options=feature['options'],
                        index=0
                    )
            
            if st.button("Predict"):
                processed_data = preprocess_data(input_data, preprocessor)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    prediction_prob = model.predict_proba(processed_data)[:, 1][0]
                    prediction_class = 1 if prediction_prob >= 0.5 else 0
                else:
                    prediction_class = model.predict(processed_data)[0]
                    prediction_prob = float(prediction_class)
                
                # Display prediction results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("Prediction Result")
                    
                    # Prediction probability gauge
                    st.metric(
                        label="Probability of Default",
                        value=f"{prediction_prob:.2%}"
                    )
                    
                    # Prediction class with colored box
                    if prediction_class == 1:
                        st.error(f"Prediction: Default (Class {prediction_class})")
                    else:
                        st.success(f"Prediction: No Default (Class {prediction_class})")
                
                with col2:
                    st.header("Prediction Explanation")
                    
                    # Calculate SHAP values
                    if explainer:
                        shap_values, feat_names = get_shap_values(model, explainer, processed_data, feature_names)
                        top_features = get_top_features(shap_values, feat_names)
                        
                        if top_features:
                            st.subheader("Top 5 Contributing Features")
                            
                            # Create a horizontal bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            features = [item[0] for item in top_features]
                            importances = [abs(item[2]) for item in top_features]
                            
                            # Determine colors based on positive/negative contribution
                            colors = ['#ff9999' if item[2] > 0 else '#66b3ff' for item in top_features]
                            
                            # Create the plot
                            y_pos = np.arange(len(features))
                            ax.barh(y_pos, importances, color=colors)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(features)
                            ax.set_xlabel('Feature Importance (SHAP value magnitude)')
                            
                            # Add a legend
                            pos_patch = mpatches.Patch(color='#ff9999', label='Increases Default Risk')
                            neg_patch = mpatches.Patch(color='#66b3ff', label='Decreases Default Risk')
                            ax.legend(handles=[pos_patch, neg_patch], loc='lower right')
                            
                            st.pyplot(fig)
                    else:
                        st.write("SHAP explanation not available for this model type.")
        
        with tabs[2]:
            st.header("Batch Prediction")
            
            # Sample CSV template with proper format
            st.subheader("CSV Format Example")
            sample_data = {}
            for feature in schema['numeric_features']:
                sample_data[feature['name']] = feature.get('default', 0.0)
            for feature in schema['categorical_features']:
                sample_data[feature['name']] = feature['options'][0]
            
            sample_df = pd.DataFrame([sample_data])
            
            # Create a download button for the sample template
            buffer = BytesIO()
            sample_df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="Download Sample CSV Template",
                data=buffer,
                file_name="sample_loan_data.csv",
                mime="text/csv"
            )
            
            # Upload CSV file
            uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(batch_data.head())
                    
                    if st.button("Run Batch Prediction"):
                        # Process data and make predictions
                        processed_batch_data = preprocess_data(batch_data, preprocessor)
                        
                        # Make predictions
                        if hasattr(model, 'predict_proba'):
                            batch_probs = model.predict_proba(processed_batch_data)[:, 1]
                            batch_classes = np.where(batch_probs >= 0.5, 1, 0)
                        else:
                            batch_classes = model.predict(processed_batch_data)
                            batch_probs = batch_classes.astype(float)
                        
                        # Add predictions to dataframe
                        results_df = batch_data.copy()
                        results_df['Default_Probability'] = batch_probs
                        results_df['Predicted_Class'] = batch_classes
                        results_df['Predicted_Label'] = [schema['target']['labels'][int(c)] for c in batch_classes]
                        
                        # Display results
                        st.subheader("Batch Prediction Results")
                        st.dataframe(results_df)
                        
                        # Add download button for results
                        results_csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=results_csv,
                            file_name="loan_default_predictions.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please make sure the model files are available in the models directory.")

if __name__ == "__main__":
    main()