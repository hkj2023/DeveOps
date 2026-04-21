import pandas as pd
import joblib

# Load the trained model (saved in RandomForest.py)
model = joblib.load("defect_prediction.pkl")

# Load new data for inference
# Example: if you want to predict on a CSV file with unseen data
new_data = pd.read_csv("ML_Final_Final.csv")   # <-- replace with your actual file

# Make predictions
predictions = model.predict(new_data)

print("Predictions:", predictions)
# Save results
output = pd.DataFrame({"predictions": predictions})
output.to_csv("inference_output.csv", index=False)

print("Inference completed. Results saved to inference_output.csv")

