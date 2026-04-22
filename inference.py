import joblib
import pandas as pd

# Load trained model
model = joblib.load("defect_prediction.pkl")

# Load raw dataset
df = pd.read_csv("ML_Final_Final.csv")

# Drop target column if exists
X = df.drop("DefectLabel", axis=1) if "DefectLabel" in df.columns else df

# Predict
predictions = model.predict(X)

# Save results
output = pd.DataFrame({"prediction": predictions})
output.to_csv("inference_output.csv", index=False)

print("Inference completed. Results saved to inference_output.csv")
