import os
import subprocess
import sys

# Set MLflow tracking URI to the local mlruns directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

# Ensure mlruns directory has right permissions
print("Setting up permissions for mlruns directory...")
mlruns_dir = "./mlruns"
os.makedirs(mlruns_dir, exist_ok=True)
os.makedirs(os.path.join(mlruns_dir, ".trash"), exist_ok=True)

# Set full permissions for mlruns directory and all subdirectories
for root, dirs, files in os.walk(mlruns_dir):
    os.chmod(root, 0o777)
    for dir in dirs:
        os.chmod(os.path.join(root, dir), 0o777)
    for file in files:
        os.chmod(os.path.join(root, file), 0o666)

# Start MLflow UI on port 7860 (HF default)
if __name__ == "__main__":
    print("Starting MLflow UI on port 7860...")
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui", 
        "--host", "0.0.0.0", 
        "--port", "7860"
    ]) 