import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Load raw dataset
df = pd.read_csv("ML_Final_Final.csv")

X = df.drop("DefectLabel", axis=1)
y = df["DefectLabel"]

# Check if model already exists
if os.path.exists("defect_prediction.pkl"):
    print("Model defect_prediction.pkl already exists. Skipping retraining.")
else:
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "defect_prediction.pkl")
    print("Training completed. Model saved to defect_prediction.pkl")
