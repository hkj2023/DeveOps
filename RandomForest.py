import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(RAW_PATH)

TARGET_COL = "TargetColumn"

# Encode target column if it's categorical
y = df[TARGET_COL].astype("category").cat.codes

# Drop target from features
X = df.drop(columns=[TARGET_COL])

# One-hot encode categorical features
X = pd.get_dummies(X)

# Save column schema
feature_names = X.columns.tolist()

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model + schema
joblib.dump((model, feature_names), MODEL_PATH)

print("Model trained and saved successfully")
