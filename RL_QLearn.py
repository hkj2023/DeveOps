import numpy as np
import matplotlib.pyplot as plt
import json

# Q-Learning parameters
states = ["LowRisk", "MediumRisk", "HighRisk"]
actions = ["TestMore", "Deploy", "Rollback"]

Q = np.zeros((len(states), len(actions)))
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
episodes = 1000

# Reward function
def reward(state, action):
    if state == "HighRisk" and action == "Deploy":
        return -10
    if state == "HighRisk" and action == "Rollback":
        return +10
    if state == "MediumRisk" and action == "TestMore":
        return +5
    if state == "LowRisk" and action == "Deploy":
        return +10
    return 0

# Training loop
for ep in range(episodes):
    state_idx = np.random.randint(0, len(states))
    action_idx = np.random.randint(0, len(actions))
    r = reward(states[state_idx], actions[action_idx])
    Q[state_idx, action_idx] = Q[state_idx, action_idx] + alpha * (
        r + gamma * np.max(Q[state_idx]) - Q[state_idx, action_idx]
    )

# Save Q-table as JSON
q_table_dict = {
    states[s]: {actions[a]: float(Q[s, a]) for a in range(len(actions))}
    for s in range(len(states))
}
with open("policy.json", "w") as f:
    json.dump(q_table_dict, f, indent=4)

# Plot Q-values
plt.figure(figsize=(8, 6))
for s in range(len(states)):
    plt.plot(actions, Q[s], marker="o", label=f"State: {states[s]}")
plt.title("Q-Learning Policy Values")
plt.xlabel("Actions")
plt.ylabel("Q-Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("q_learning_policy.png")   # save as artifact
plt.close()

print("RL Q-Learning complete. Policy saved as policy.json and plot saved as q_learning_policy.png")
