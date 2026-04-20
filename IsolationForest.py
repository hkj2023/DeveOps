# ==========================================
# Unsupervised Defect/Anomaly Detection with Isolation Forest
# ==========================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("ML_Final_Final.csv")

# 2. Encode categorical variables (Isolation Forest needs numeric input)
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 3. Select features for anomaly detection
# You can include all numeric columns or a subset relevant to CI/CD
feature_cols = df.select_dtypes(include=[np.number]).columns
X = df[feature_cols]

# 4. Train Isolation Forest
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # expected proportion of anomalies (adjust as needed)
    random_state=42
)
iso.fit(X)

# 5. Predict anomalies
df["Anomaly"] = iso.predict(X)   # -1 = anomaly, 1 = normal
df["AnomalyFlag"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

# 6. Save results
df.to_csv("ML_unsupervised_isolation.csv", index=False)
print("✅ Isolation Forest anomaly detection complete. Results saved as ML_unsupervised_isolation.csv")

# 7. Summary
summary = df["AnomalyFlag"].value_counts()
print(summary)

# ==========================================
# 🔥 NEW: Create JSON for Test Orchestration
# ==========================================

import json

total = int(len(df))
anomalies = int(summary.get("Anomaly", 0))
normal = int(summary.get("Normal", 0))

anomaly_ratio = float(anomalies / total) if total > 0 else 0.0

# Decision logic
decision = {
    "anomaly_detected": bool(anomalies > 0),
    "anomaly_count": anomalies,
    "total_records": total,
    "anomaly_ratio": round(anomaly_ratio, 4),
    "action": "",
    "trigger_alert": False
}

# Rules
if anomaly_ratio > 0.1:
    decision["action"] = "run_full_tests"
    decision["trigger_alert"] = True

elif anomaly_ratio > 0.02:
    decision["action"] = "run_priority_tests"

else:
    decision["action"] = "normal_pipeline"

# 🔥 FINAL SAFETY CONVERSION (very important)
decision = {k: (int(v) if isinstance(v, np.integer)
                else float(v) if isinstance(v, np.floating)
                else bool(v) if isinstance(v, np.bool_)
                else v)
            for k, v in decision.items()}

# Save JSON
with open("anomaly_decision.json", "w") as f:
    json.dump(decision, f, indent=4)

print("✅ Anomaly decision JSON created successfully!")