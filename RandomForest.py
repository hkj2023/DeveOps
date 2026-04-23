import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "outputs/defect_prediction.pkl"

# Skip training if model already exists
if os.path.exists(MODEL_PATH):
    print(f"Model already exists at {MODEL_PATH}. Skipping training.")
else:
    # Load dataset
    df = pd.read_csv("ML_Final_Final.csv")
    print("Dataset loaded with shape:", df.shape)

    # Create DefectLabel from DefectCount if missing
    if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
        df["DefectLabel"] = (df["DefectCount"] > 500).astype(int)

    # Separate features and target
    X = df.drop("DefectLabel", axis=1)
    y = df["DefectLabel"]

    # Convert categorical/string columns to numeric (one-hot encoding)
    X = pd.get_dummies(X)

    # Define the model with resource‑friendly parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)

    # Save model + feature names
    os.makedirs("outputs", exist_ok=True)
    joblib.dump((model, list(X.columns)), MODEL_PATH)
    print(f"Training completed. Model saved to {MODEL_PATH}")
