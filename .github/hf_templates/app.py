import os
import subprocess
import sys

# Set MLflow tracking URI to the local mlruns directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

# Start MLflow UI on port 7860 (HF default)
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui", 
        "--host", "0.0.0.0", 
        "--port", "7860"
    ]) 