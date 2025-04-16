import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib

# Set MLflow tracking URI - using a local directory
mlflow.set_tracking_uri("file:./mlruns")

# Create experiment
experiment_name = "classification_experiment"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

# Create synthetic data with make_classification
def generate_data(n_samples=1000, n_features=20, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=10, 
        n_redundant=5, 
        n_classes=2, 
        random_state=random_state
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs("./data", exist_ok=True)
    joblib.dump(scaler, "./data/scaler.joblib")
    
    # Save the test data for later evaluation
    np.save("./data/X_test.npy", X_test)
    np.save("./data/y_test.npy", y_test)
    
    # Also save the scaled test data
    np.save("./data/X_test_scaled.npy", X_test_scaled)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Training function with MLflow tracking
def train_and_log_model(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run():
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train the model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Create and log feature importance plot
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            features = [f"Feature {i}" for i in range(X_train.shape[1])]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title('Feature Importances')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig("feature_importances.png")
            mlflow.log_artifact("feature_importances.png")
            plt.close()
        
        return model, accuracy, roc_auc

if __name__ == "__main__":
    # Generate data
    X_train, X_test, y_train, y_test = generate_data()
    
    # Define parameter grid for hyperparameter tuning
    param_grid = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": 42},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42},
        {"n_estimators": 150, "max_depth": None, "min_samples_split": 10, "random_state": 42},
        {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "random_state": 42},
        {"n_estimators": 250, "max_depth": 10, "min_samples_split": 5, "random_state": 42},
    ]
    
    # Train models with different parameters and log to MLflow
    print("Training models with different hyperparameters...")
    best_model = None
    best_roc_auc = 0
    best_params = None
    
    for params in param_grid:
        print(f"Training with parameters: {params}")
        model, accuracy, roc_auc = train_and_log_model(X_train, X_test, y_train, y_test, params)
        
        # Keep track of the best model
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_params = params
    
    print("\nTraining completed!")
    print(f"Best model parameters: {best_params}")
    print(f"Best model ROC AUC: {best_roc_auc:.4f}")
    
    # Register the best model in MLflow Model Registry
    with mlflow.start_run():
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log best metrics
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Register the model
        mlflow.sklearn.log_model(
            best_model, 
            "best_model",
            registered_model_name="BestClassificationModel"
        )
        
        # Save the best model locally
        os.makedirs("./models", exist_ok=True)
        joblib.dump(best_model, "./models/best_model.joblib")
        
        print("\nBest model registered as 'BestClassificationModel' in MLflow Model Registry")
        print("Best model also saved locally at './models/best_model.joblib'") 