import joblib
import pandas as pd

# Load model and feature names
model, feature_names = joblib.load("defect_prediction.pkl")

df = pd.read_csv("ML_Final_Final.csv")

# Create DefectLabel if missing
if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
    df["DefectLabel"] = (df["DefectCount"] > 500).astype(int)

# Apply same encoding
X = df.drop("DefectLabel", axis=1)
X = pd.get_dummies(X)

# Align with training features (fill missing with 0)
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]

predictions = model.predict(X)

output = pd.DataFrame({"prediction": predictions})
output.to_csv("inference_output.csv", index=False)

print("Inference completed. Results saved to inference_output.csv")
