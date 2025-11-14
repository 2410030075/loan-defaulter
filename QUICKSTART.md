# Quick Start Guide

This guide provides step-by-step instructions to quickly get started with the Loan Default Prediction project.

## 1. Clone the Repository (if needed)

```powershell
git clone <repository-url>
cd loan-default-project
```

## 2. Set up the Environment

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Run the End-to-End Pipeline

### Option 1: Step-by-Step

```powershell
# Preprocess data
python src/data_preprocessing.py

# Train models
python src/train_model.py

# Evaluate models
python src/evaluate.py

# Launch Streamlit app
streamlit run src/app_streamlit.py
```

### Option 2: Direct Use

If models are already trained, you can directly launch the Streamlit app:

```powershell
streamlit run src/app_streamlit.py
```

## 4. Using the Streamlit App

1. Access the app at http://localhost:8501

2. **Single Prediction Tab**
   - Fill out the form with loan applicant details
   - Click "Predict" to get the default probability
   - View SHAP feature importance explanations
   - Adjust the prediction threshold as needed

3. **Batch Prediction Tab**
   - Upload a CSV file with multiple loan applications
   - Review the predictions table
   - Download results as a CSV file

4. **Model Insights Tab**
   - Explore feature importance charts
   - View performance metrics
   - Understand the model's decision patterns

## 5. Interpreting Results

- Default Probability: Higher values indicate higher risk of default
- SHAP Values: Show how each feature contributes to the prediction
- Red features: Increase default probability
- Blue features: Decrease default probability

## Troubleshooting

1. **Missing models error**: Make sure you've run `train_model.py` first
2. **Import errors**: Verify all dependencies are installed with `pip install -r requirements.txt`
3. **Data path issues**: Ensure the data file is in the correct location (`data/Loan_Default.csv`)