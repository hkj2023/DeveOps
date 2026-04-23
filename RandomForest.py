import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

RAW_PATH = "ML_Final_Final.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

os.makedirs("outputs", exist_ok=True)

# Load raw dataset
df = pd.read_csv(RAW_PATH)

# Encode features
df_clean = pd.get_dummies(df.dropna())

X = df_clean.drop("TargetColumn", axis=1)   # replace with your target column
y = df_clean["TargetColumn"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and feature names
joblib.dump((model, X.columns.tolist()), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
