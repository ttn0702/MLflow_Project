import os
import sys
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_generation():
    """Test that the data generation works correctly."""
    X, y = make_classification(
        n_samples=100, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        n_classes=2, 
        random_state=42
    )
    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert set(np.unique(y)) == {0, 1}
    
def test_data_preprocessing():
    """Test the data preprocessing pipeline."""
    # Generate some sample data
    X, y = make_classification(
        n_samples=100, 
        n_features=20, 
        random_state=42
    )
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check shapes
    assert X_train_scaled.shape == (80, 20)
    assert X_test_scaled.shape == (20, 20)
    
    # Check scaling properties
    assert np.isclose(np.mean(X_train_scaled), 0, atol=1e-10)
    assert np.isclose(np.std(X_train_scaled), 1, atol=0.1)

def test_model_training():
    """Test that the model can be trained and makes reasonable predictions."""
    # Generate and preprocess data
    X, y = make_classification(
        n_samples=100, 
        n_features=20, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    model = RandomForestClassifier(
        n_estimators=10,  # Small number for quick testing
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Check predictions
    assert y_pred.shape == (20,)
    assert y_proba.shape == (20,)
    
    # Check performance is better than random
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    assert accuracy > 0.5  # Should be better than random
    assert roc_auc > 0.5   # Should be better than random

if __name__ == "__main__":
    # For manual test running
    test_data_generation()
    test_data_preprocessing()
    test_model_training()
    print("All tests passed!") 