import os
import pandas as pd
import joblib

MODEL_PATH = "outputs/defect_prediction.pkl"
DATA_PATH = "outputs/new_data.csv"        # correct path inside outputs
OUTPUT_PATH = "outputs/inference_output.csv"

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")

# Ensure new data exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"New data file not found at {DATA_PATH}. Run prep first.")

# Load model and features
model, feature_names = joblib.load(MODEL_PATH)

# Load new unseen data
df = pd.read_csv(DATA_PATH)

# Align columns with training features
X = df.reindex(columns=feature_names, fill_value=0)

# Run predictions
predictions = model.predict(X)

# Save predictions
pd.DataFrame({"Prediction": predictions}).to_csv(OUTPUT_PATH, index=False)

print(f"Predictions saved to {OUTPUT_PATH}")
