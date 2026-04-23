import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

# Load raw dataset
df = pd.read_csv(RAW_PATH)

# Separate target column (replace 'TargetColumn' with your actual label column)
y = df["TargetColumn"]
X = df.drop("TargetColumn", axis=1)

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# Save model and feature names
joblib.dump((model, X_encoded.columns.tolist()), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
