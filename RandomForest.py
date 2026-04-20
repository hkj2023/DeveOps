# ==========================================
# Defect Prediction with Random Forest + JSON Export
# ==========================================

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("ML_Final_Final.csv")

# -----------------------------
# 2. Define target column (binary classification)
# -----------------------------
df["DefectLabel"] = (df["DefectDensity"] > 0.5).astype(int)
print("Class distribution:\n", df["DefectLabel"].value_counts())

# -----------------------------
# 3. Encode categorical variables
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------
# 4. Split features and target
# -----------------------------
X = df.drop(columns=["DefectCount", "DefectDensity", "DefectLabel"])
y = df["DefectLabel"]

# -----------------------------
# 5. Balance dataset with SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# -----------------------------
# 6. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# -----------------------------
# 7. Random Forest + Hyperparameter Tuning
# -----------------------------
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = best_rf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_prob = best_rf.predict_proba(X_test)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -----------------------------
# 9. Save Trained Model
# -----------------------------
joblib.dump(best_rf, "defect_prediction_rf.pkl")
print("✅ Model saved as defect_prediction_rf.pkl")

# -----------------------------
# 10. Generate Predictions JSON for Jenkins
# -----------------------------
# Reload model (optional, ensures consistency)
model = joblib.load("defect_prediction_rf.pkl")

# Use same preprocessed dataset
X_full = df.drop(columns=["DefectCount", "DefectDensity", "DefectLabel"], errors="ignore")

# Predict probabilities (risk scores)
risk_scores = model.predict_proba(X_full)[:, 1]

# Build dictionary of module → risk score
identifier_col = "ModuleName" if "ModuleName" in df.columns else None
if identifier_col:
    predictions = {
        row[identifier_col]: float(score)
        for row, score in zip(df.to_dict(orient="records"), risk_scores)
    }
else:
    predictions = {f"Row_{i}": float(score) for i, score in enumerate(risk_scores)}

# Save to JSON
with open("defect_predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

print("✅ Defect predictions exported to defect_predictions.json")
