import os
import pandas as pd
import subprocess

def test_prep_creates_csv():
    # Run prep.py
    subprocess.run(["python", "prep.py"], check=True)
    # Check output file exists
    assert os.path.exists("outputs/new_data.csv")
    # Check file has expected columns
    df = pd.read_csv("outputs/new_data.csv")
    assert "feature1" in df.columns  # replace with actual feature names
