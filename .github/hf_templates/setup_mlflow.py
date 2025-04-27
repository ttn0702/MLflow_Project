import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI - using a local directory
mlflow.set_tracking_uri("file:./mlruns")

def setup_mlflow():
    logger.info("Setting up MLflow environment...")
    
    # Create classification experiment if it doesn't exist
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
    
    # Check if model is already registered
    try:
        models = mlflow.search_registered_models(filter_string=f"name='BestClassificationModel'")
        if models and len(models) > 0:
            logger.info("BestClassificationModel already exists in the registry")
            return
    except Exception as e:
        logger.warning(f"Error checking registered models: {e}")
    
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
    
    logger.info("MLflow setup completed successfully")

if __name__ == "__main__":
    try:
        setup_mlflow()
    except Exception as e:
        logger.error(f"Error during MLflow setup: {e}", exc_info=True) 