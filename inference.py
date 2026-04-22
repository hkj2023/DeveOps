import joblib
import pandas as pd

model = joblib.load("defect_prediction.pkl")
df = pd.read_csv("ML_Final_Final.csv")

X = df.drop("DefectLabel", axis=1) if "DefectLabel" in df.columns else df
predictions = model.predict(X)

output = pd.DataFrame({"prediction": predictions})
output.to_csv("inference_output.csv", index=False)

print("Inference completed. Results saved to inference_output.csv")
