# defect_prediction.py
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
# Replace 'ML_Final_Final.csv' with your actual dataset path
df = pd.read_csv("ML_Final_Final.csv")

# 2. Create target label if not already present
if "DefectLabel" not in df.columns and "DefectCount" in df.columns:
    df["DefectLabel"] = (df["DefectCount"] > 0).astype(int)

# 3. Separate features and target
X = df.drop(columns=["DefectLabel"], errors="ignore")
y = df["DefectLabel"]

# Convert categorical features to numeric (one-hot encoding)
X = pd.get_dummies(X)

# 4. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation metrics
results = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

# 8. Save metrics to JSON
with open("outputs/defect_prediction_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Defect prediction completed. Results:")
print(results)
