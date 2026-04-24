import subprocess

def run_step(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}:\n{result.stderr}")
        raise SystemExit(result.returncode)
    else:
        print(result.stdout)

if __name__ == "__main__":
    run_step("RandomForest.py")   # Train model → saves defect_prediction.pkl
    run_step("prep.py")           # Prepare new_data.csv
    run_step("inference.py")      # Run predictions
    print("✅ Evaluation pipeline completed successfully.")

