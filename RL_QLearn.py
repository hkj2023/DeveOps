# ==========================================
# Reinforcement Learning (Q-Learning) for Pipeline Optimization
# ==========================================

import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("ML_Final_Final.csv")

# 2. Define states (discretize numeric features including Coverage %)
df["CommitRisk_bin"]   = pd.qcut(df["CommitRisk"], q=5, labels=False, duplicates="drop")
df["Coverage_bin"]     = pd.qcut(df["Coverage %"], q=5, labels=False, duplicates="drop")
df["FailureRate_bin"]  = pd.qcut(df["FailureRate"], q=5, labels=False, duplicates="drop")

states = list(zip(df["CommitRisk_bin"], df["Coverage_bin"], df["FailureRate_bin"]))
print("Unique FailureRate values:", df["FailureRate"].nunique())

# 🔥 FIX 1: remove duplicate states
unique_states = list(set(states))

# 3. Define actions
actions = ["increase_tests", "reduce_commit_size", "rerun_failed_tests"]

# 4. Initialize Q-table
Q = {s: {a: 0.0 for a in actions} for s in unique_states}

# 5. Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

# 6. Reward function
def get_reward(row, action):
    if action == "increase_tests":
        return row["Coverage %"] * 0.1 - row["FailureRate"]
    elif action == "reduce_commit_size":
        return -row["CommitRisk"] * 0.05 + (1 - row["FailureRate"])
    elif action == "rerun_failed_tests":
        return -row["FailureRate"] + 0.5
    return 0

# 7. Q-Learning loop
for ep in range(episodes):
    row = df.sample(1).iloc[0]
    state = (row["CommitRisk_bin"], row["Coverage_bin"], row["FailureRate_bin"])

    # epsilon-greedy
    if random.uniform(0,1) < epsilon:
        action = random.choice(actions)
    else:
        action = max(Q[state], key=Q[state].get)

    reward = get_reward(row, action)

    old_value = Q[state][action]
    next_state = state  # simplified environment
    next_max = max(Q[next_state].values())

    Q[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

# 8. Extract optimal policy
policy = {s: max(Q[s], key=Q[s].get) for s in Q}

# 9. Safe lookup
def best_action(state, policy, default="rerun_failed_tests"):
    return policy.get(state, default)

# ==========================================
# 🔥 FIX 2: Convert tuple keys → reys for JSON
# ==========================================
policy_json = {f"{s[0]}_{s[1]}_{s[2]}": a for s, a in policy.items()}

with open("policy.json", "w") as f:
    json.dump(policy_json, f, indent=4)

# ==========================================
# OPTIONAL: Load back (for reuse)
# ==========================================
with open("policy.json", "r") as f:
    loaded_policy_json = json.load(f)

policy_loaded = {tuple(map(int, k.split("_"))): v for k, v in loaded_policy_json.items()}

# 10. Demonstration
print("Q-Learning complete. Optimal pipeline actions learned.")
print("Sample policy (state,  best action):")

for k, v in list(policy.items())[:10]:
    print(k, " ", v)

# Example usage
test_states = [(4, 3, 3), (3, 2, 1), (2, 0, 0)]
for s in test_states:
    print(f"State {s},  Action: {best_action(s, policy_loaded)}")

# 11. Visualization
action_counts = pd.Series(list(policy.values())).value_counts()

plt.figure(figsize=(6,4))
action_counts.plot(kind="bar", color=["#4CAF50", "#2196F3", "#FF9800"])
plt.title("Optimal Policy Action Distribution")
plt.xlabel("Action")
plt.ylabel("Frequency")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 12. Export counts
action_counts.to_csv("policy_action_distribution.csv", header=["Frequency"])

