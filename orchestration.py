import json

with open("defect_predictions.json") as f:
    defects = json.load(f)
with open("anomaly_decision.json") as f:
    anomalies = json.load(f)
with open("policy.json") as f:
    policy = json.load(f)

print("Defect Predictions:", defects)
print("Anomaly Decisions:", anomalies)
print("RL Policy:", policy)
