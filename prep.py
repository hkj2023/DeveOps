import pandas as pd

df = pd.read_csv("ML_Final_Final.csv")
df = df.dropna()
df = pd.get_dummies(df)
df.to_csv("processed_data.csv", index=False)

print("Data preprocessing completed. Results saved to processed_data.csv")
