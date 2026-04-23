import pandas as pd
import os

RAW_PATH = "ML_Final_Final.csv"
PROCESSED_PATH = "outputs/processed_data.csv"
NEW_DATA_PATH = "outputs/new_data.csv"

os.makedirs("outputs", exist_ok=True)

# Load raw dataset
df = pd.read_csv(RAW_PATH)

# Example preprocessing: drop NA, encode categorical
df_clean = pd.get_dummies(df.dropna())

# Save processed dataset for training
df_clean.to_csv(PROCESSED_PATH, index=False)

# Simulate fresh unseen data (subset or modified rows)
df_new = df_clean.sample(min(10, len(df_clean)), random_state=42)
df_new.to_csv(NEW_DATA_PATH, index=False)

print(f"Processed data saved to {PROCESSED_PATH}")
print(f"New unseen data saved to {NEW_DATA_PATH}")
