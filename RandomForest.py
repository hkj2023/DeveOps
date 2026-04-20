import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("ML_Final_Final.csv")

print("COLUMNS IN DATASET:")
print(df.columns.astype(str))

# Ensure target column exists
if "DefectLabel" not in df.columns:
    print("DefectLabel not found → creating from DefectCount")
    df["DefectLabel"] = (df["DefectCount"] > 0).astype(int)

# Encode categorical variables (compatible with Pandas 2/3)
categorical_cols = df.select_dtypes(include=["object", "string"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separate features and target
X = df.drop("DefectLabel", axis=1)
y = df["DefectLabel"]

print("Class distribution:")
print(y.value_counts())

# Handle class imbalance with SMOTE only if >1 class
if len(y.unique()) > 1:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Class distribution after resampling:")
    print(y_resampled.value_counts())
else:
    print("SMOTE skipped: only one class present")
    X_resampled, y_resampled = X, y

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Define model and hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predictions
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Save model
joblib.dump(best_rf, "random_forest_model.pkl")

# Save predictions to JSON
predictions = {
    "y_true": y_test.tolist(),
    "y_pred": y_pred.tolist(),
    "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(y.unique()) > 1 else None
}
with open("defect_predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

# Reports (plain text, no emoji)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
if len(y.unique()) > 1:
    print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

