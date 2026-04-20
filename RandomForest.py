import sys
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Fix encoding (CI/CD safe)
# -----------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv("ML_Final_Final.csv")

# -----------------------------
# 3. Clean column names
# -----------------------------
df.columns = df.columns.str.strip()

print("COLUMNS IN DATASET:")
print(df.columns)

# -----------------------------
# 4. CREATE TARGET COLUMN (IMPORTANT FIX)
# -----------------------------
if "DefectLabel" not in df.columns:
    if "DefectCount" in df.columns:
        print("\nDefectLabel not found → creating from DefectCount")
        df["DefectLabel"] = (df["DefectCount"] > 0).astype(int)
    else:
        raise ValueError("No valid target column found (DefectLabel or DefectCount)")

target_col = "DefectLabel"

# -----------------------------
# 5. Handle missing values
# -----------------------------
df = df.fillna(0)

# -----------------------------
# 6. Encode categorical variables
# -----------------------------
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -----------------------------
# 7. Split features & target
# -----------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

# -----------------------------
# 8. Handle imbalance
# -----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass distribution after SMOTE:")
print(y_resampled.value_counts())

# -----------------------------
# 9. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 10. Model training
# -----------------------------
rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)

# -----------------------------
# 11. Predictions
# -----------------------------
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# -----------------------------
# 12. Save model
# -----------------------------
joblib.dump(best_rf, "random_forest_model.pkl")

# -----------------------------
# 13. Save results
# -----------------------------
results = {
    "y_true": y_test.tolist(),
    "y_pred": y_pred.tolist(),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}

with open("defect_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# 14. Evaluation
# -----------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
