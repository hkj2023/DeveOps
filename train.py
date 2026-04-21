# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your preprocessed data
# Make sure prep.py outputs a file like data.csv with features + labels
data = pd.read_csv("ML_Final_Final.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
with open("defect_prediction.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete. Model saved as defect_prediction.pkl")
