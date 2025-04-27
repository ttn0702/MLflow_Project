import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI - using a local directory
mlflow.set_tracking_uri("file:./mlruns")

def setup_mlflow():
    logger.info("Setting up MLflow environment...")
    
    # Quick check for existing data
    if has_existing_data():
        logger.info("Found existing MLflow data. No need to create sample data.")
        return
    
    # If we reached here, there's no existing data, so we'll create sample data
    logger.info("No existing MLflow data found. Creating sample data...")
    create_sample_data()
    
    logger.info("MLflow setup completed successfully")

def has_existing_data():
    # Check if there are any existing experiments besides Default
    all_experiments = mlflow.search_experiments()
    if len(all_experiments) > 1:  # More than just the Default experiment
        return True
    
    # Check if there are any existing runs
    experiment_dirs = glob.glob("./mlruns/[0-9]*")
    for exp_dir in experiment_dirs:
        run_dirs = glob.glob(f"{exp_dir}/*/")
        if run_dirs:
            return True
    
    # Check if models registry contains any models
    try:
        models = mlflow.search_registered_models()
        if models and len(models) > 0:
            return True
    except Exception as e:
        logger.warning(f"Error checking registered models: {e}")
    
    return False

def create_sample_data():
    # Create classification experiment
    experiment_name = "classification_experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Create and log a simple model
    logger.info("Creating and registering sample classification model...")
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    
    # Generate some dummy data for the model to fit
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Fit the model
    model.fit(X, y)
    
    # Log the model to MLflow
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_param("n_estimators", 10)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("precision", 0.83)
        mlflow.log_metric("recall", 0.82)
        mlflow.log_metric("f1_score", 0.82)
        mlflow.log_metric("roc_auc", 0.90)
        
        # Log model
        logger.info("Logging model to MLflow")
        mlflow.sklearn.log_model(
            model, 
            "sample_model",
            registered_model_name="BestClassificationModel"
        )

if __name__ == "__main__":
    try:
        setup_mlflow()
    except Exception as e:
        logger.error(f"Error during MLflow setup: {e}", exc_info=True) 