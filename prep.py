import pandas as pd

# Load raw dataset
df = pd.read_csv("ML_Final_Final.csv")

# Example preprocessing
df = df.dropna()
df = pd.get_dummies(df)

# Save processed data (optional, for inspection)
df.to_csv("processed_data.csv", index=False)

print("Prep stage completed. Processed data saved to processed_data.csv")
