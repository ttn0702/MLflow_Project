from flask import Flask, request, jsonify, render_template, Markup
import numpy as np
import joblib
import os
import sys
import markdown
import codecs

# Add the parent directory to the path so we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Load the model from local file
def load_model():
    try:
        # First try to load model from the current directory structure
        model_path = os.path.join("models", "best_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
            return model
            
        # If not found, try alternative paths
        alternative_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.joblib"),
            "/app/models/best_model.joblib"  # Docker container path
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"Loaded model from {path}")
                return model
                
        raise FileNotFoundError(f"Model file not found in any of the expected locations: {[model_path] + alternative_paths}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the scaler
def load_scaler():
    try:
        # First try to load scaler from the current directory structure
        scaler_path = os.path.join("data", "scaler.joblib")
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
            
        # If not found, try alternative paths
        alternative_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scaler.joblib"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scaler.joblib"),
            "/app/data/scaler.joblib"  # Docker container path
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                return joblib.load(path)
                
        raise FileNotFoundError(f"Scaler file not found in any of the expected locations: {[scaler_path] + alternative_paths}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        raise

# Load model and scaler on startup
try:
    model = load_model()
    scaler = load_scaler()
    print("Successfully loaded model and scaler")
except Exception as e:
    print(f"Failed to load model or scaler: {e}")
    # Don't raise exception here, let the app start anyway
    # We'll handle the error in the endpoints

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    # Path to the ProjectDescription.md file
    doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ProjectDescription.md")
    
    # Nếu file không tồn tại, thử các đường dẫn khác
    if not os.path.exists(doc_path):
        alternative_paths = [
            "ProjectDescription.md",  # Thử tìm ở thư mục hiện tại
            os.path.join("app", "ProjectDescription.md"),  # Thử tìm trong thư mục app
            "/app/ProjectDescription.md",  # Thư mục gốc trong Docker
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProjectDescription.md")  # Thư mục của app.py
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                doc_path = path
                break
    
    try:
        # Nếu tìm thấy file, đọc nó
        if os.path.exists(doc_path):
            with codecs.open(doc_path, mode="r", encoding="utf-8") as f:
                text = f.read()
            
            # Convert markdown to HTML
            html = markdown.markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])
        else:
            # Nếu không tìm thấy file, hiển thị thông báo đơn giản
            html = """
            <h1>MLflow Project Documentation</h1>
            <p>Tài liệu đầy đủ không khả dụng, nhưng bạn có thể xem tài liệu trực tuyến tại 
            <a href="https://github.com/yourusername/MLflow_Project/blob/main/ProjectDescription.md" target="_blank">GitHub Repository</a>.</p>
            
            <h2>Tính năng chính:</h2>
            <ul>
                <li>Huấn luyện mô hình phân loại nhị phân với RandomForest</li>
                <li>Sử dụng MLflow theo dõi thí nghiệm</li>
                <li>Tìm siêu tham số tối ưu</li>
                <li>Cung cấp API dự đoán</li>
                <li>Giao diện web thân thiện</li>
            </ul>
            
            <h2>Cách sử dụng</h2>
            <p>Nhập 20 giá trị đặc trưng và nhấn nút <strong>Predict</strong> để nhận kết quả phân loại.</p>
            """
    except Exception as e:
        html = f"<h1>Documentation Not Available</h1><p>Error: {str(e)}</p>"
    
    # Pass the HTML to the template using Markup to prevent escaping
    return render_template('documentation.html', content=Markup(html))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and scaler are loaded
        if 'model' not in globals() or 'scaler' not in globals():
            raise RuntimeError("Model or scaler not loaded properly")
            
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