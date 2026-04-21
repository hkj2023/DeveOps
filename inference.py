import joblib
import pandas as pd

# Load trained model
model = joblib.load("defect_prediction.pkl")

# Load preprocessed input data
df = pd.read_csv("processed_data.csv")

# Drop target column if exists
if "DefectLabel" in df.columns:
    X = df.drop("DefectLabel", axis=1)
else:
    X = df

# Predict
predictions = model.predict(X)

# Save results
output = pd.DataFrame({"prediction": predictions})
output.to_csv("inference_output.csv", index=False)

print("Inference completed. Results saved to inference_output.csv")
