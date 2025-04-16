from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import sys
import mlflow
import mlflow.sklearn

# Add the parent directory to the path so we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Load the model from MLflow or from the local file
def load_model():
    try:
        # Try to load from MLflow model registry
        mlflow.set_tracking_uri("file:../mlruns")
        model = mlflow.sklearn.load_model("models:/BestClassificationModel/latest")
        print("Loaded model from MLflow model registry")
    except Exception as e:
        print(f"Failed to load model from MLflow registry: {e}")
        # Fallback to local model file
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib")
        model = joblib.load(model_path)
        print("Loaded model from local file")
    
    return model

# Load the scaler
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scaler.joblib")
    return joblib.load(scaler_path)

# Load model and scaler on startup
model = load_model()
scaler = load_scaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            # If JSON data
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
        else:
            # If form data
            features = []
            for i in range(20):  # Assuming 20 features as in the training data
                feature_val = request.form.get(f'feature_{i}', 0)
                features.append(float(feature_val))
            features = np.array(features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability)
        }
        
        # If the request is from a form, render a template
        if not request.is_json:
            return render_template('result.html', prediction=result['prediction'], 
                                  probability=round(result['probability'] * 100, 2))
        
        # Otherwise return JSON
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"), exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000) 