# MLflow Classification Project

This project demonstrates a simple MLflow implementation for a classification task using synthetic data. It showcases model training, hyperparameter tuning, model registry, and deployment through a Flask web application.

## Project Structure

```
MLflow_Project/
|-- data/              # Directory for storing generated data
|-- models/            # Directory for storing trained models
|-- app/               # Flask web application
|   |-- templates/     # HTML templates for the web app
|   |   |-- index.html
|   |   |-- result.html
|   |-- app.py         # Flask application code
|-- train.py           # Main script for data generation and model training
|-- run.py             # Helper script to run training, MLflow UI, and Flask app
|-- requirements.txt   # Project dependencies
|-- README.md          # This file
```

## Requirements

This project requires Python 3.8+ and the following packages:
- scikit-learn
- pandas
- numpy
- mlflow
- flask
- gunicorn
- matplotlib
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

The project includes a helper script (`run.py`) that simplifies running different components:

```bash
# Run model training
python run.py train

# Run the Flask web application
python run.py app

# Run the MLflow UI 
python run.py ui

# Run everything in sequence (training, MLflow UI, then Flask app)
python run.py all
```

### Manual Usage

If you prefer to run components individually:

#### 1. Training Models and Hyperparameter Tuning

Run the training script to:
- Generate synthetic classification data
- Train several RandomForest classifiers with different hyperparameters
- Track models and metrics with MLflow
- Register the best model in the MLflow Model Registry

```bash
python train.py
```

#### 2. Running the Web Application

After training, start the Flask application to serve predictions using the best model:

```bash
cd app
python app.py
```

The web application will be accessible at: http://localhost:5000

#### 3. Using MLflow UI

You can explore the training results using the MLflow UI:

```bash
mlflow ui
```

This will start the MLflow UI at http://localhost:5000 (make sure the Flask app is not running at the same port).

## Model Information

The project uses a RandomForest classifier for a binary classification task. Hyperparameter tuning focuses on:
- Number of estimators
- Maximum depth
- Minimum samples split

Performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC score

## Web Application

The Flask web application provides a simple interface to:
- Input feature values (20 features)
- Get classification predictions from the best model
- View prediction probabilities 