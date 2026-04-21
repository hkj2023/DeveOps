import joblib
import pandas as pd
import os

# =========================
# Load trained model
# =========================
model_path = "defect_prediction.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)
print("Model loaded successfully.")

# =========================
# Load processed data
# =========================
data_path = "processed_data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Processed data file not found: {data_path}")

df = pd.read_csv(data_path)
print(f"Processed data loaded. Shape: {df.shape}")

# =========================
# Prepare features
# =========================
if "DefectLabel" in df.columns:
    X = df.drop("DefectLabel", axis=1)
    print("Dropped target column 'DefectLabel' from input.")
else:
    X = df
    print("No target column found. Using full dataset for inference.")

# =========================
# Run prediction
# =========================
predictions = model.predict(X)

# =========================
# Save output
# =========================
output = pd.DataFrame({
    "prediction": predictions
})

output_path = "inference_output.csv"
output.to_csv(output_path, index=False)

print(f"Inference completed successfully. Results saved to {output_path}")
