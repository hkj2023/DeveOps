import pandas as pd

# Load raw data
df = pd.read_csv("ML_Final_Final.csv")

# Example preprocessing: drop missing values
df = df.dropna()

# Example feature engineering: one-hot encode categorical variables
df = pd.get_dummies(df)

# Save processed data
df.to_csv("processed_data.csv", index=False)

print("Data preprocessing completed. Results saved to processed_data.csv")
