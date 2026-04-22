import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

df = pd.read_csv("ML_Final_Final.csv")

# Create target if missing
if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
    df["DefectLabel"] = (df["DefectCount"] > 500).astype(int)

X = df.drop("DefectLabel", axis=1)
y = df["DefectLabel"]

model = RandomForestClassifier()
model.fit(X, y)

# Save both model and feature names together
joblib.dump((model, list(X.columns)), "defect_prediction.pkl")

print("Training completed. Model + feature names saved to defect_prediction.pkl")
