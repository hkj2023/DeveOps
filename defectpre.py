# defect_prediction.py
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# 1. Load dataset
df = pd.read_csv("Final.csv")  # replace with your actual file

# 2. Create target label from DefectCount
df["DefectLabel"] = (df["DefectCount"] > 0).astype(int)

# 3. Feature engineering from TestsRun and TestsFailed
df["FailureRatio"] = df["TestsFailed"] / df["TestsRun"].replace(0, 1)  # avoid division by zero
df["SuccessRate"] = (df["TestsRun"] - df["TestsFailed"]) / df["TestsRun"].replace(0, 1)
df["HasFailure"] = (df["TestsFailed"] > 0).astype(int)

# 4. Select features (drop DefectCount to avoid leakage)
features = [
    "FailureRatio", "SuccessRate", "HasFailure",
    "Coverage %", "CommitRisk", "FilesChanged",
    "LinesAdded", "LinesRemoved", "FailureSeverityIndex"
]

X = df[features]
y = df["DefectLabel"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# Safe probability extraction
if len(model.classes_) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of defect
else:
    y_prob = model.predict_proba(X_test)[:, 0]  # only one class present

# 8. Evaluation metrics
results_split = {
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0)
}

# 9. Cross-validation for more robust evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

results_cv = {
    "cv_f1_scores": cv_scores.tolist(),
    "cv_f1_mean": cv_scores.mean()
}

# 10. Save metrics and predictions
output = {
    "split_metrics": results_split,
    "cross_validation": results_cv,
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist()
}

with open("outputs/prediction_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Defect prediction completed. Results:")
print(json.dumps(output, indent=2))
