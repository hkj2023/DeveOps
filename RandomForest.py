# ==========================================
# Defect Prediction with Random Forest
# ==========================================

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(RAW_PATH)

# -----------------------------
# 2. Create binary target from DefectCount
# -----------------------------
# Rule: DefectLabel = 1 if DefectCount > 400, else 0
df["DefectLabel"] = (df["DefectCount"] > 400).astype(int)

y = df["DefectLabel"]

# -----------------------------
# 3. Prepare features
# -----------------------------
# Drop DefectCount and DefectLabel from features
X = df.drop(columns=["DefectCount", "DefectLabel"])

# One-hot encode categorical features (critical fix)
X_encoded = pd.get_dummies(X)

# Save schema
feature_names = X_encoded.columns.tolist()

# -----------------------------
# 4. Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

# -----------------------------
# 5. Save model + schema
# -----------------------------
joblib.dump((model, feature_names), MODEL_PATH)

print("✅ Model trained and saved successfully")
