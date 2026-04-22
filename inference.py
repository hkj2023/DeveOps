import os
import joblib
import pandas as pd

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load model and feature names
model, feature_names = joblib.load("outputs/defect_prediction.pkl")

# Load dataset
df = pd.read_csv("ML_Final_Final.csv")

# Create DefectLabel if missing
if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
    df["DefectLabel"] = (df["DefectCount"] > 500).astype(int)

# Prepare features (one-hot encode categorical)
X = df.drop("DefectLabel", axis=1)
X = pd.get_dummies(X)

# Align with training features
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]

# Run predictions
predictions = model.predict(X)

# Build output DataFrame
output = pd.DataFrame({"prediction": predictions})

# Save results
output.to_csv("outputs/inference_output.csv", index=False)

print("Inference completed. Results saved to outputs/inference_output.csv")
