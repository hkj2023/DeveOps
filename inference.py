import os
import pandas as pd

OUTPUT_PATH = "outputs/inference_output.csv"

# Ensure predictions file exists
if not os.path.exists(OUTPUT_PATH):
    raise FileNotFoundError(f"Inference output file not found at {OUTPUT_PATH}. Run inference stage first.")

# Load predictions
df = pd.read_csv(OUTPUT_PATH)

print(f"Loaded predictions from {OUTPUT_PATH}:")
print(df.head())   # show first few rows

