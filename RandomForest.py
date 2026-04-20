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

# ✅ Ensure UTF-8 encoding (fixes Windows CI issues)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("ML_Final_Final.csv")

# -----------------------------
# 2. Handle missing values (important for stability)
# -----------------------------
df = df.fillna(0)

# -----------------------------
# 3. Encode categorical variables (Pandas 2/3 safe)
# -----------------------------
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -----------------------------
# 4. Separate features & target
# -----------------------------
if "DefectLabel" not in df.columns:
    raise ValueError("Target column 'DefectLabel' not found in dataset")

X = df.drop("DefectLabel", axis=1)
y = df["DefectLabel"]

# -----------------------------
# 5. Handle class imbalance
# -----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after resampling:")
print(y_resampled.value_counts())

# -----------------------------
# 6. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Model & hyperparameter tuning
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

print("Best Parameters:", grid_search.best_params_)

# -----------------------------
# 8. Predictions
# -----------------------------
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# -----------------------------
# 9. Save model
# -----------------------------
joblib.dump(best_rf, "random_forest_model.pkl")

# -----------------------------
# 10. Save predictions (safe JSON)
# -----------------------------
predictions = {
    "y_true": y_test.tolist(),
    "y_pred": y_pred.tolist(),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}

with open("defect_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4)

# -----------------------------
# 11. Evaluation (NO emojis)
# -----------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
