import joblib
import pandas as pd

# Load model and feature names
model, feature_names = joblib.load("defect_prediction.pkl")

df = pd.read_csv("ML_Final_Final.csv")

# Create target if missing
if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
    df["DefectLabel"] = (df["DefectCount"] > 500).astype(int)

# Align dataset with training features
X = df[feature_names]

predictions = model.predict(X)

output = pd.DataFrame({"prediction": predictions})
output.to_csv("inference_output.csv", index=False)

print("Inference completed. Results saved to inference_output.csv")
