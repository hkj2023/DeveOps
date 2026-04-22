import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("ML_Final_Final.csv")

# Create DefectLabel from DefectCount if missing
if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
    df["DefectLabel"] = (df["DefectCount"] > 500).astype(int)

# Separate features and target
X = df.drop("DefectLabel", axis=1)
y = df["DefectLabel"]

# Convert categorical/string columns to numeric (one-hot encoding)
X = pd.get_dummies(X)

# Train model
# model = RandomForestClassifier()
# model.fit(X, y)

# Define the model with resource‑friendly parameters
model = RandomForestClassifier(
    n_estimators=100,   # fewer trees
    max_depth=20,       # limit depth
    n_jobs=-1,          # use all cores
    random_state=42
)
# Train model
model.fit(X, y)

# Save model + feature names
joblib.dump((model, list(X.columns)), "outputs/defect_prediction.pkl")
print("Training completed. Model and feature names saved to defect_prediction.pkl")
