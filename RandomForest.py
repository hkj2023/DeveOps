import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(RAW_PATH)

TARGET_COL = "TargetColumn"

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# ✅ FORCE consistent encoding (IMPORTANT FIX)
X = pd.get_dummies(X)

# Save column schema FIRST (critical)
feature_names = X.columns.tolist()

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump((model, feature_names), MODEL_PATH)

print("Model trained and saved successfully")
