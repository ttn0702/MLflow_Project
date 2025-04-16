import os
import sys
import subprocess
import time
import webbrowser

def run_training():
    """Run the model training script"""
    print("Starting model training and hyperparameter tuning...")
    subprocess.run([sys.executable, "train.py"])
    print("Training completed!")

def run_mlflow_ui():
    """Run the MLflow UI"""
    print("Starting MLflow UI...")
    # Run MLflow UI in the background
    process = subprocess.Popen([
        sys.executable, "-m", "mlflow", "ui", "--port", "5001"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:5001")
    return process

def run_flask_app():
    """Run the Flask web application"""
    print("Starting Flask web application...")
    os.chdir("app")
    # Run Flask app in the foreground
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            run_training()
        elif command == "app":
            run_flask_app()
        elif command == "ui":
            mlflow_process = run_mlflow_ui()
            print("Press Ctrl+C to stop the MLflow UI")
            try:
                # Keep the script running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                mlflow_process.terminate()
                print("MLflow UI stopped")
        elif command == "all":
            run_training()
            mlflow_process = run_mlflow_ui()
            run_flask_app()
            mlflow_process.terminate()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, app, ui, all")
    else:
        print("Please specify a command: train, app, ui, or all")
        print("  train: Run model training")
        print("  app:   Run Flask web application")
        print("  ui:    Run MLflow UI")
        print("  all:   Run training, start MLflow UI, and run the Flask app") 