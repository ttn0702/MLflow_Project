from flask import Flask, request, jsonify, render_template, Markup
import numpy as np
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
import markdown
import codecs

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

@app.route('/documentation')
def documentation():
    # Path to the ProjectDescription.md file
    doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ProjectDescription.md")
    
    # Read the markdown file
    with codecs.open(doc_path, mode="r", encoding="utf-8") as f:
        text = f.read()
    
    # Convert markdown to HTML
    html = markdown.markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])
    
    # Pass the HTML to the template using Markup to prevent escaping
    return render_template('documentation.html', content=Markup(html))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            # If JSON data
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
            
            # Scale the features
            scaled_features = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0][1]
            
            # Return JSON response
            return jsonify({
                'prediction': int(prediction),
                'probability': float(probability)
            })
        else:
            # If form data
            features = []
            for i in range(20):  # Assuming 20 features as in the training data
                feature_val = request.form.get(f'feature_{i}', 0)
                features.append(float(feature_val))
            
            # Create features array for display
            feature_values = [float(request.form.get(f'feature_{i}', 0)) for i in range(20)]
            
            # Scale the features
            features_array = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0][1]
            
            # Return to index with prediction results
            return render_template('index.html', 
                                  has_prediction=True,
                                  prediction=int(prediction), 
                                  probability=round(probability * 100, 2),
                                  feature_values=feature_values)
    
    except Exception as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"), exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000) 