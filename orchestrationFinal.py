import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def orchestrate(defect_file, anomaly_file, policy_file, output_file="risk_classification.json"):
    # Load artifacts
    defects = load_json(defect_file)       # {'y_true': [...], 'y_pred': [...]}
    anomalies = load_json(anomaly_file)    # {'AnomalyFlag': {...}, 'decisions': [...]}
    policy = load_json(policy_file)        # {'state_action_map': {...}}

    risk_results = []

    # Iterate over predictions and anomalies together
    for idx, (y_true, y_pred) in enumerate(zip(defects["y_true"], defects["y_pred"])):
        anomaly_flag = anomalies["decisions"][idx]  # 0 = Normal, 1 = Anomaly

        if y_pred == 1 and anomaly_flag == 1:
            risk = "High Risk"
        elif y_pred == 0 and anomaly_flag == 1:
            risk = "Medium Risk"
        elif y_pred == 1 and anomaly_flag == 0:
            risk = "Medium Risk (Defect only)"
        else:
            risk = "Low Risk"

        # Optional: attach RL policy recommendation if state available
        state_key = str(idx)  # or however you map states
        action = policy.get("state_action_map", {}).get(state_key, "N/A")

        risk_results.append({
            "module_id": idx,
            "defect_pred": y_pred,
            "anomaly_flag": anomaly_flag,
            "risk_level": risk,
            "recommended_action": action
        })

    # Save unified output
    with open(output_file, "w") as f:
        json.dump(risk_results, f, indent=2)

    print(f"Unified risk classification saved to {output_file}")

if __name__ == "__main__":
    orchestrate("defect_predictions.json", "anomaly_decision.json", "policy.json")
