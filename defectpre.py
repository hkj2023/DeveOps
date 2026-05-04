# defect_prediction_fixed.py
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("your_dataset.csv")  # replace with your actual file

# 2. Create target label from DefectCount
df["DefectLabel"] = (df["DefectCount"] > 0).astype(int)

# 3. Engineer features from TestsRun and TestsFailed
df["FailureRatio"] = df["TestsFailed"] / df["TestsRun"].replace(0, 1)
df["SuccessRate"] = (df["TestsRun"] - df["TestsFailed"]) / df["TestsRun"].replace(0, 1)
df["HasFailure"] = (df["TestsFailed"] > 0).astype(int)

# 4. Select only indirect predictors (drop leakage features)
features = [
    "FailureRatio", "SuccessRate", "HasFailure",
    "Coverage %", "CommitRisk", "FilesChanged",
    "LinesAdded", "LinesRemoved", "FailureSeverityIndex"
]
X = df[features]
y = df["DefectLabel"]

# 5. Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
if len(model.classes_) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = model.predict_proba(X_test)[:, 0]

# Split metrics
results_split = {
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring="f1")
results_cv = {"cv_f1_scores": cv_scores.tolist(), "cv_f1_mean": cv_scores.mean()}

# Save output
output = {
    "split_metrics": results_split,
    "cross_validation": results_cv,
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist()
}

with open("outputs/defect_prediction_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
