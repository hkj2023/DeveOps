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
    # Step 1: Prepare new unseen data
    run_step("prep.py")

    # Step 2: Run inference on prepared data
    run_step("inference.py")

    print("✅ Evaluation pipeline completed successfully.")
