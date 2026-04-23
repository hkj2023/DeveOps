import pandas as pd, os

RAW_PATH = "ML_Final_Final.csv"
PROCESSED_PATH = "outputs/processed_data.csv"
NEW_DATA_PATH = "outputs/new_data.csv"

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(RAW_PATH)
df_clean = pd.get_dummies(df.dropna())
df_clean.to_csv(PROCESSED_PATH, index=False)

# Create fresh unseen data for inference
df_new = df_clean.sample(min(10, len(df_clean)), random_state=42)
df_new.to_csv(NEW_DATA_PATH, index=False)

print(f"Processed data saved to {PROCESSED_PATH}")
print(f"New unseen data saved to {NEW_DATA_PATH}")
