import os
import subprocess
import sys

# Set MLflow tracking URI to point directly to the mlruns folder
MLFLOW_DIR = os.path.abspath("./mlruns")
print(f"Using MLflow tracking directory: {MLFLOW_DIR}")

# Ensure the mlruns directory exists
if not os.path.exists(MLFLOW_DIR):
    print(f"Creating MLflow directory: {MLFLOW_DIR}")
    os.makedirs(MLFLOW_DIR, exist_ok=True)

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = f"file:{MLFLOW_DIR}"
print(f"Set MLFLOW_TRACKING_URI to: {os.environ['MLFLOW_TRACKING_URI']}")

# Disable MLflow DB migrations which can cause permission issues
os.environ["MLFLOW_DISABLE_DB_MIGRATIONS"] = "true"

# Start MLflow UI on port 7860 (HF default)
if __name__ == "__main__":
    print("Starting MLflow UI on port 7860...")
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui", 
        "--host", "0.0.0.0", 
        "--port", "7860",
        "--no-serve-artifacts"  # Avoid additional permission issues with artifact serving
    ]) 