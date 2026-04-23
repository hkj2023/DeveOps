import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv(RAW_PATH)

# Create binary target from DefectCount > 400
df["DefectLabel"] = (df["DefectCount"] > 400).astype(int)
y = df["DefectLabel"]

# Drop target + raw DefectCount
X = df.drop(columns=["DefectCount", "DefectLabel"])

# --- FIX: Encode categorical columns ---
# Detect non-numeric columns automatically
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# Save schema (feature names)
feature_names = X_encoded.columns.tolist()

# Train model on encoded features
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# Save model + schema together
joblib.dump({"model": model, "features": feature_names}, MODEL_PATH)

print("Model trained and saved successfully")
