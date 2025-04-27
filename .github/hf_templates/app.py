import os
import subprocess
import sys
import tempfile
import shutil

# Create a temporary directory for MLflow
print("Creating temporary directory for MLflow tracking...")
temp_mlruns_dir = tempfile.mkdtemp(prefix="mlflow_temp_")
print(f"Temporary MLflow directory: {temp_mlruns_dir}")

# Set MLflow tracking URI to the temporary directory
os.environ["MLFLOW_TRACKING_URI"] = f"file:{temp_mlruns_dir}"

# Copy existing experiment data if available
if os.path.exists("./mlruns") and os.path.isdir("./mlruns"):
    print("Copying existing MLflow data to temporary directory...")
    try:
        # Copy only experiment directories, not .trash
        for item in os.listdir("./mlruns"):
            if item != ".trash" and os.path.isdir(os.path.join("./mlruns", item)):
                src = os.path.join("./mlruns", item)
                dst = os.path.join(temp_mlruns_dir, item)
                shutil.copytree(src, dst)
        print("Existing data copied successfully")
    except Exception as e:
        print(f"Warning: Could not copy existing data: {e}")

# Start MLflow UI on port 7860 (HF default)
if __name__ == "__main__":
    print("Starting MLflow UI on port 7860...")
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui", 
        "--host", "0.0.0.0", 
        "--port", "7860"
    ]) 