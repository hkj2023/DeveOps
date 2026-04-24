import os
import subprocess
import pandas as pd

def test_inference_creates_predictions():
    # Run inference script
    subprocess.run(["python", "inference.py"], check=True)
    # Check predictions file exists
    assert os.path.exists("outputs/inference_output.csv")
    # Check predictions file has content
    df = pd.read_csv("outputs/inference_output.csv")
    assert "Prediction" in df.columns
    assert len(df) > 0
