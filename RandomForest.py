import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = "outputs/processed_data.csv"
MODEL_PATH = "outputs/defect_prediction.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Run prep first.")

# Load processed data
df = pd.read_csv(DATA_PATH)

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + feature names
joblib.dump((model, X.columns.tolist()), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
