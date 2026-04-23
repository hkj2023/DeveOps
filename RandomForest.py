import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv(RAW_PATH)

# ⚠️ Replace with your actual target column name
TARGET_COL = "TargetColumn"

# Split features and target
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# ✅ Convert ALL categorical columns to numeric
X = pd.get_dummies(X)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model + feature names for inference alignment
joblib.dump((model, X.columns.tolist()), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
