import os
import subprocess
import sys

# Set the project root directory to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print("Running Loan Default Prediction Project...")
print("1. Data Preprocessing")
result = subprocess.run(["python", "src/data_preprocessing.py"], check=True)

print("\n2. Training Models")
try:
    result = subprocess.run(["python", "src/train_model.py"], check=True)
except subprocess.CalledProcessError:
    print("Error in model training. Modifying train_model.py...")
    
    # Fix the import statement in train_model.py
    with open("src/train_model.py", "r") as file:
        content = file.read()
    
    # Replace the import statement
    content = content.replace(
        "from src.data_preprocessing import preprocess_and_save",
        "from src.data_preprocessing import preprocess_data"
    )
    
    # Also update the function call
    content = content.replace(
        "X_train, X_test, y_train, y_test, preprocessor = preprocess_and_save()",
        "X_train, X_test, y_train, y_test = preprocess_data()"
    )
    
    with open("src/train_model.py", "w") as file:
        file.write(content)
    
    print("Fixed train_model.py, trying again...")
    result = subprocess.run(["python", "src/train_model.py"], check=True)

print("\n3. Evaluating Models")
try:
    result = subprocess.run(["python", "src/evaluate.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error in model evaluation: {e}")

print("\nProject execution complete. You can now launch the Streamlit app:")
print("streamlit run src/app_streamlit.py")