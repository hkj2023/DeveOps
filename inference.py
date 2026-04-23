
import os
import pandas as pd
import joblib

MODEL_PATH = "outputs/defect_prediction.pkl"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")

# Load model and feature names
model, feature_names = joblib.load(MODEL_PATH)

# Load new data for prediction
df = pd.read_csv("new_data.csv")   # mount this file at runtime
X = pd.get_dummies(df)

# Align columns with training features
X = X.reindex(columns=feature_names, fill_value=0)

# Run predictions
predictions = model.predict(X)
print("Predictions:", predictions)
