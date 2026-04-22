import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("ML_Final_Final.csv")

# Drop non-numeric columns (IDs, strings)
df = df.drop(columns=["CommitID", "BuildID", "TestID"], errors='ignore')

# Separate features and target
X = df.drop("DefectLabel", axis=1)
y = df["DefectLabel"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("defect_prediction.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")
