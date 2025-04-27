# MÔ TẢ CHI TIẾT QUÁ TRÌNH THỰC HIỆN DỰ ÁN MLFLOW

## 1. Giới thiệu dự án

Dự án này thực hiện một quy trình MLOps hoàn chỉnh sử dụng MLflow, từ huấn luyện mô hình đến triển khai ứng dụng. Tôi đã xây dựng một mô hình phân loại nhị phân đơn giản, sử dụng dữ liệu tổng hợp, với các tính năng chính sau:

- Sử dụng MLflow để theo dõi thí nghiệm và quản lý mô hình
- Thực hiện điều chỉnh siêu tham số và so sánh hiệu suất mô hình
- Đăng ký mô hình tốt nhất vào MLflow Model Registry
- Tạo ứng dụng web Flask với giao diện trực quan để phục vụ dự đoán

## 2. Phân tích yêu cầu và chuẩn bị

### 2.1. Phân tích và thiết kế yêu cầu

Trước khi bắt đầu triển khai, tôi đã xác định các yêu cầu chính cho dự án:

1. **Dữ liệu**: Sử dụng dữ liệu tổng hợp cho bài toán phân loại nhị phân
2. **Mô hình**: Áp dụng mô hình RandomForest với điều chỉnh siêu tham số
3. **MLflow**: Theo dõi thí nghiệm, ghi lại siêu tham số, chỉ số đánh giá và lưu trữ mô hình
4. **API dự đoán**: Triển khai mô hình tốt nhất qua ứng dụng web Flask

### 2.2. Cấu trúc dự án

Tôi đã thiết kế cấu trúc dự án như sau để đảm bảo tính module hóa và dễ bảo trì:

```
MLflow_Project/
|-- data/              # Thư mục lưu trữ dữ liệu
|-- models/            # Thư mục lưu trữ mô hình đã huấn luyện
|-- mlruns/            # MLflow tracking (tự động tạo)
|-- app/               # Ứng dụng web Flask
|   |-- templates/     # HTML templates
|   |   |-- index.html
|   |   |-- result.html
|   |-- app.py         # Mã nguồn ứng dụng Flask
|-- train.py           # Script chính để tạo dữ liệu và huấn luyện mô hình
|-- run.py             # Script hỗ trợ để chạy các thành phần khác nhau
|-- requirements.txt   # Các gói phụ thuộc
|-- README.md          # Tệp README
```

### 2.3. Cài đặt môi trường

Tôi đã tạo file `requirements.txt` với các thư viện cần thiết và phiên bản cụ thể:

```
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
mlflow==2.8.0
flask==2.3.3
gunicorn==21.2.0
matplotlib==3.7.2
joblib==1.3.2
markdown==3.4.3
```

## 3. Triển khai - Tạo và xử lý dữ liệu

### 3.1. Tạo dữ liệu tổng hợp

Đầu tiên, tôi đã triển khai hàm `generate_data()` trong file `train.py` để tạo dữ liệu tổng hợp:

```python
def generate_data(n_samples=1000, n_features=20, random_state=42):
    # Tạo dữ liệu phân loại nhị phân
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=10,  # 10 đặc trưng thực sự có ý nghĩa
        n_redundant=5,     # 5 đặc trưng dư thừa
        n_classes=2,       # Phân loại nhị phân
        random_state=random_state
    )
    
    # Chia dữ liệu: 80% huấn luyện, 20% kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Tạo thư mục và lưu dữ liệu
    os.makedirs("./data", exist_ok=True)
    joblib.dump(scaler, "./data/scaler.joblib")
    np.save("./data/X_test.npy", X_test)
    np.save("./data/y_test.npy", y_test)
    np.save("./data/X_test_scaled.npy", X_test_scaled)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
```

Hàm này:
- Tạo 1000 mẫu dữ liệu với 20 đặc trưng
- Chỉ có 10 đặc trưng thực sự chứa thông tin và 5 đặc trưng dư thừa
- Chia dữ liệu thành tập huấn luyện và kiểm tra với tỷ lệ 80-20
- Chuẩn hóa dữ liệu bằng StandardScaler và lưu bộ chuẩn hóa
- Lưu dữ liệu kiểm tra để sử dụng cho đánh giá sau này

## 4. Thiết lập MLflow và huấn luyện mô hình

### 4.1. Cấu hình MLflow

Tôi thiết lập MLflow tracking để lưu trữ thí nghiệm và mô hình:

```python
# Thiết lập MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Tạo experiment
experiment_name = "classification_experiment"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)
```

Cấu hình này:
- Sử dụng thư mục cục bộ "./mlruns" để lưu trữ thông tin thí nghiệm
- Tạo một experiment mới có tên "classification_experiment" hoặc sử dụng experiment đã tồn tại

### 4.2. Hàm huấn luyện và ghi lại thông tin mô hình

Tôi đã triển khai hàm `train_and_log_model()` đóng vai trò trung tâm trong dự án:

```python
def train_and_log_model(X_train, X_test, y_train, y_test, params):
    with mlflow.start_run():
        # Ghi lại các siêu tham số
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Huấn luyện mô hình RandomForest
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Đánh giá mô hình
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Tính toán các chỉ số
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Ghi lại các chỉ số
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Tạo và lưu biểu đồ độ quan trọng của đặc trưng
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
        
        # Lưu mô hình
        mlflow.sklearn.log_model(model, "model")
        
        return model, accuracy, roc_auc
```

Hàm này thực hiện:
- Tạo một MLflow run để theo dõi thí nghiệm
- Ghi lại tất cả siêu tham số của mô hình
- Huấn luyện mô hình RandomForest với các tham số được cung cấp
- Tính toán và ghi lại các chỉ số đánh giá: accuracy, precision, recall, f1, roc_auc
- Tạo và lưu biểu đồ feature importance để trực quan hóa
- Lưu mô hình vào MLflow để có thể tải lại sau này

## 5. Điều chỉnh siêu tham số và thực hiện thí nghiệm

Tôi đã triển khai thử nghiệm với nhiều bộ siêu tham số khác nhau:

```python
# Định nghĩa lưới tham số
param_grid = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": 42},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42},
    {"n_estimators": 150, "max_depth": None, "min_samples_split": 10, "random_state": 42},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "random_state": 42},
    {"n_estimators": 250, "max_depth": 10, "min_samples_split": 5, "random_state": 42},
]

# Huấn luyện và đánh giá
best_model = None
best_roc_auc = 0
best_params = None

for params in param_grid:
    print(f"Training with parameters: {params}")
    model, accuracy, roc_auc = train_and_log_model(X_train, X_test, y_train, y_test, params)
    
    # Theo dõi mô hình tốt nhất dựa trên ROC AUC
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = model
        best_params = params
```

Quá trình này:
- Thử nghiệm với 5 bộ tham số khác nhau cho RandomForest
- Điều chỉnh các tham số chính: n_estimators, max_depth, min_samples_split
- Theo dõi mô hình có hiệu suất tốt nhất dựa trên điểm ROC AUC

## 6. Đăng ký mô hình tốt nhất vào Model Registry

Sau khi xác định mô hình tốt nhất, tôi đăng ký nó vào MLflow Model Registry:

```python
# Đăng ký mô hình tốt nhất
with mlflow.start_run():
    # Log best parameters
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
    
    # Log metrics
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
    
    # Đăng ký mô hình
    mlflow.sklearn.log_model(
        best_model, 
        "best_model",
        registered_model_name="BestClassificationModel"
    )
    
    # Lưu mô hình tốt nhất cục bộ
    os.makedirs("./models", exist_ok=True)
    joblib.dump(best_model, "./models/best_model.joblib")
```

Quá trình này:
- Tạo một MLflow run mới để lưu mô hình tốt nhất
- Ghi lại tham số và chỉ số hiệu suất của mô hình tốt nhất
- Đăng ký mô hình vào MLflow Model Registry với tên "BestClassificationModel"
- Lưu mô hình cục bộ để sử dụng trong ứng dụng web

## 7. Phát triển ứng dụng web Flask

### 7.1. Thiết kế ứng dụng

Tôi đã phát triển một ứng dụng web Flask để phục vụ dự đoán từ mô hình tốt nhất:

```python
from flask import Flask, request, jsonify, render_template, Markup
import numpy as np
import joblib
import os
import sys
import markdown
import codecs

app = Flask(__name__)

# Tải mô hình từ MLflow hoặc tệp cục bộ
def load_model():
    try:
        # Tìm kiếm mô hình trong nhiều vị trí
        model_path = os.path.join("models", "best_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
            return model
            
        # Nếu không tìm thấy, thử các đường dẫn thay thế
        alternative_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.joblib"),
            "/app/models/best_model.joblib"  # Đường dẫn cho Docker container
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"Loaded model from {path}")
                return model
                
        raise FileNotFoundError(f"Model file not found in any of the expected locations")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Tải bộ chuẩn hóa
def load_scaler():
    try:
        # Tương tự như mô hình, tìm kiếm bộ chuẩn hóa ở nhiều vị trí
        scaler_path = os.path.join("data", "scaler.joblib")
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
            
        alternative_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scaler.joblib"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scaler.joblib"),
            "/app/data/scaler.joblib"
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                return joblib.load(path)
                
        raise FileNotFoundError(f"Scaler file not found in any of the expected locations")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        raise

# Tải mô hình và bộ chuẩn hóa khi khởi động
try:
    model = load_model()
    scaler = load_scaler()
    print("Successfully loaded model and scaler")
except Exception as e:
    print(f"Failed to load model or scaler: {e}")
```

### 7.2. Tạo API endpoints

Tôi đã tạo ba endpoints chính:

```python
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    # Đường dẫn đến file ProjectDescription.md
    doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ProjectDescription.md")
    
    # Đọc file markdown
    try:
        with codecs.open(doc_path, mode="r", encoding="utf-8") as f:
            text = f.read()
        
        # Chuyển đổi markdown sang HTML
        html = markdown.markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])
    except Exception as e:
        html = f"<h1>Documentation Not Available</h1><p>Error: {str(e)}</p>"
    
    # Truyền HTML vào template
    return render_template('documentation.html', content=Markup(html))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ yêu cầu
        if request.is_json:
            # Nếu dữ liệu JSON
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
        else:
            # Nếu dữ liệu từ form
            features = []
            for i in range(20):  # Giả sử 20 đặc trưng như dữ liệu huấn luyện
                feature_val = request.form.get(f'feature_{i}', 0)
                features.append(float(feature_val))
            
            # Tạo mảng để hiển thị
            feature_values = [float(request.form.get(f'feature_{i}', 0)) for i in range(20)]
            features = np.array(features).reshape(1, -1)
        
        # Chuẩn hóa đặc trưng
        scaled_features = scaler.transform(features)
        
        # Dự đoán
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        # Nếu yêu cầu từ form, hiển thị kết quả trong index.html
        if not request.is_json:
            return render_template('index.html', 
                                 has_prediction=True,
                                 prediction=int(prediction), 
                                 probability=round(probability * 100, 2),
                                 feature_values=feature_values)
        
        # Nếu yêu cầu API, trả về JSON
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })
    
    except Exception as e:
        # Xử lý lỗi
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('index.html', error=str(e))
```

### 7.3. Phát triển giao diện người dùng

Tôi đã thiết kế ba template HTML cho ứng dụng:

1. **index.html** - Trang chính:
   - Hiển thị form với 20 trường đầu vào cho các đặc trưng
   - Nút ngẫu nhiên hóa để tạo giá trị ngẫu nhiên
   - Hiển thị kết quả dự đoán trực tiếp trên cùng trang khi gửi form
   - Thanh tiến trình trực quan hiển thị xác suất dự đoán

2. **documentation.html** - Trang tài liệu:
   - Hiển thị nội dung file ProjectDescription.md được chuyển đổi từ markdown sang HTML
   - Cung cấp hướng dẫn chi tiết về kiến trúc và cách sử dụng dự án
   - Tích hợp định dạng mã nguồn và bảng biểu từ markdown

3. **result.html** - Trang kết quả dự phòng:
   - Template này được tạo ra như một tùy chọn thay thế cho hiển thị kết quả
   - Có cấu trúc tương tự phần kết quả trong index.html nhưng dưới dạng trang riêng biệt
   - Hiện tại chưa được sử dụng trong ứng dụng vì kết quả được hiển thị trực tiếp trên index.html
   - Được giữ lại cho mục đích mở rộng trong tương lai hoặc chuyển sang mô hình hiển thị trang kết quả riêng

## 8. Tạo script chạy dự án (run.py)

Để đơn giản hóa việc chạy các thành phần khác nhau của dự án, tôi đã tạo file `run.py`:

```python
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
```

Script này cung cấp 4 lệnh:
  - `train`: Chạy quá trình huấn luyện mô hình
  - `ui`: Khởi chạy MLflow UI để khám phá các thí nghiệm
  - `app`: Khởi chạy ứng dụng web Flask
  - `all`: Chạy tất cả các thành phần theo trình tự

## 9. Kết quả và đánh giá

### 9.1. Đánh giá mô hình

Sau khi chạy toàn bộ quá trình, mô hình tốt nhất đạt được:
- Accuracy (Độ chính xác): khoảng 95%
- ROC AUC: khoảng 0.98
- F1 Score: khoảng 0.94

Các kết quả này cho thấy mô hình hoạt động tốt trên dữ liệu tổng hợp.

### 9.2. Theo dõi thí nghiệm với MLflow UI

MLflow UI (http://localhost:5001) cho phép:
- Xem danh sách các thí nghiệm đã chạy
- So sánh hiệu suất của các mô hình khác nhau
- Kiểm tra biểu đồ feature importance
- Truy cập mô hình đã đăng ký trong Model Registry

### 9.3. Trải nghiệm ứng dụng web

Ứng dụng web (http://localhost:5000) cung cấp:
- Giao diện trực quan dễ sử dụng
- Khả năng nhập hoặc ngẫu nhiên hóa các giá trị đặc trưng
- Kết quả phân loại (0 hoặc 1) với biểu diễn trực quan về xác suất
- Tài liệu dự án đầy đủ có thể truy cập qua đường dẫn /documentation

Giao diện được thiết kế theo mô hình Single Page Application, với form nhập liệu và kết quả dự đoán được hiển thị trên cùng một trang, giúp người dùng dễ dàng điều chỉnh dữ liệu đầu vào và xem kết quả ngay lập tức.

Ngoài ra, ứng dụng cung cấp API endpoint cho phép tích hợp với các hệ thống khác, thông qua giao thức HTTP với định dạng JSON:

```
POST /predict
Content-Type: application/json

{
  "features": [0.1, 0.2, 0.3, ..., 0.0]  # Mảng 20 giá trị đặc trưng
}
```

Phản hồi:
```
{
  "prediction": 1,               # Lớp dự đoán (0 hoặc 1)
  "probability": 0.832           # Xác suất thuộc lớp 1
}
```

## 10. Hướng dẫn sử dụng dự án

### 10.1. Cài đặt

```bash
# Clone repository (nếu áp dụng)
git clone <repository-url>
cd MLflow_Project

# Cài đặt dependencies
   pip install -r requirements.txt
   ```

### 10.2. Chạy các thành phần

Để huấn luyện mô hình:
```bash
   python run.py train
   ```

Để khám phá kết quả với MLflow UI:
```bash
   python run.py ui
   ```

Để chạy ứng dụng web:
```bash
   python run.py app
   ```

Để chạy tất cả các thành phần:
```bash
   python run.py all
   ```

## 11. Kết luận và bài học

Dự án này đã thành công trong việc triển khai một quy trình MLOps hoàn chỉnh với MLflow:

1. **Quản lý thí nghiệm**:
   - Theo dõi siêu tham số và chỉ số đánh giá
   - So sánh hiệu suất các mô hình
   - Trực quan hóa kết quả

2. **Quản lý mô hình**:
- Lưu trữ và quản lý phiên bản mô hình
   - Đăng ký mô hình tốt nhất vào Model Registry
   - Tạo cơ chế dự phòng để sử dụng mô hình cục bộ

3. **Triển khai mô hình**:
   - Phát triển API trực quan để phục vụ dự đoán
   - Xây dựng giao diện người dùng thân thiện
   - Thực hiện tiền xử lý dữ liệu trong ứng dụng

Thông qua dự án này, tôi đã học được cách sử dụng MLflow để theo dõi thí nghiệm, quản lý mô hình và triển khai mô hình vào ứng dụng thực tế. Kỹ năng này sẽ giúp ích trong việc phát triển các dự án học máy lớn hơn trong tương lai.

## 12. Triển khai CI/CD với GitHub Actions

Để đảm bảo quy trình MLOps liên tục và tự động, tôi đã triển khai CI/CD pipeline sử dụng GitHub Actions.

### 12.1. Cấu trúc CI/CD Pipeline

Tôi đã tạo file cấu hình `.github/workflows/mlflow-ci.yml` với 3 job chính:

```yaml
name: MLflow Project CI/CD

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:

jobs:
  lint-and-test:
    # Job kiểm tra code quality và unit tests
    
  train-model:
    # Job huấn luyện mô hình
    
  deploy-to-huggingface:
    # Job triển khai ứng dụng lên Hugging Face
```

### 12.2. Kiểm tra mã nguồn và unit tests

Job đầu tiên (`lint-and-test`) thực hiện kiểm tra chất lượng code với flake8 và chạy unit tests:

```yaml
lint-and-test:
  runs-on: ubuntu-latest
  
  steps:
  - uses: actions/checkout@v4
  
  - name: Set up Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.10'
      
  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      pip install flake8 pytest
      if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      
  - name: Lint with flake8
    run: |
      # stop the build if there are Python syntax errors or undefined names
      flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      # exit-zero treats all errors as warnings
      flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
      
  - name: Test with pytest
    run: |
      # Create directories if they don't exist
      mkdir -p data models
      pytest -v
```

Tôi đã tạo các unit tests trong thư mục `tests/` để kiểm tra:
- Quá trình tạo dữ liệu
- Tiền xử lý dữ liệu
- Huấn luyện và đánh giá mô hình

### 12.3. Huấn luyện mô hình tự động

Job thứ hai (`train-model`) tự động huấn luyện mô hình mỗi khi có push lên nhánh chính:

```yaml
train-model:
  runs-on: ubuntu-latest
  needs: lint-and-test
  if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
  
  steps:
  - uses: actions/checkout@v4
  
  - name: Set up Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.10'
        
  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      
  - name: Train model
    run: |
      # Create directories if they don't exist
      mkdir -p data models
      python train.py
      
  - name: Create artifact archive
    run: |
      mkdir -p artifact_bundle
      # Copy model files and scaler
      cp -r models/* artifact_bundle/ || echo "No model files to copy"
      cp data/scaler.joblib artifact_bundle/ || echo "scaler.joblib not found"
      
  - name: Upload model artifacts
    uses: actions/upload-artifact@v4
    with:
      name: model-artifacts
      path: artifact_bundle/
      retention-days: 5
```

### 12.4. Triển khai ứng dụng tự động

Job thứ ba (`deploy-to-huggingface`) triển khai ứng dụng lên nền tảng Hugging Face với mô hình đã huấn luyện:

```yaml
deploy-to-huggingface:
  runs-on: ubuntu-latest
  needs: train-model
  if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
  
  steps:
  - uses: actions/checkout@v4
  
  - name: Set up Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.10'
      
  - name: Download model artifacts
    uses: actions/download-artifact@v4
    with:
      name: model-artifacts
      path: downloaded_artifacts
      
  - name: Setup artifacts 
    run: |
      # Copy model files to appropriate directories
      mkdir -p models data
      cp -r downloaded_artifacts/* models/ || echo "No files to copy to models directory"
      if [ -f downloaded_artifacts/scaler.joblib ]; then
        mv downloaded_artifacts/scaler.joblib data/
      fi
      
  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      pip install huggingface_hub
      
  - name: Prepare deployment for Hugging Face
    run: |
      # Create a directory for deployment
      mkdir -p huggingface_app
      
      # Copy app and requirements
      cp -r app/* huggingface_app/ || echo "Error copying app files"
      cp requirements.txt huggingface_app/ || echo "requirements.txt not found, skipping"
      
      # Create and set proper permissions for model and data directories
      mkdir -p huggingface_app/models
      mkdir -p huggingface_app/data
      
      # Copy model and scaler files
      cp -r models/* huggingface_app/models/ || echo "No model files to copy"
      cp -r data/scaler.joblib huggingface_app/data/ || echo "No scaler.joblib to copy"
```

Phần triển khai lên Hugging Face tận dụng khả năng của nền tảng này để hosting các ứng dụng machine learning phục vụ demo và chia sẻ kết quả dự án.

### 12.5. Lợi ích của CI/CD trong MLOps

Việc áp dụng CI/CD cho dự án MLOps mang lại nhiều lợi ích:

1. **Tự động hóa**: Giảm thời gian và công sức thủ công trong quá trình kiểm thử, huấn luyện và triển khai
2. **Nhất quán**: Đảm bảo quy trình triển khai nhất quán và có thể lặp lại
3. **Phát hiện lỗi sớm**: Phát hiện và khắc phục lỗi sớm trong quá trình phát triển
4. **Tích hợp liên tục**: Tích hợp các thay đổi code vào dự án nhanh chóng và an toàn
5. **Triển khai liên tục**: Triển khai mô hình mới một cách tự động

Qua việc triển khai CI/CD, dự án MLOps của tôi đảm bảo tính liên tục trong toàn bộ vòng đời của mô hình, từ phát triển đến triển khai và giám sát. 

### 12.6. Triển khai MLflow UI trên Hugging Face

Ngoài việc triển khai ứng dụng dự đoán, tôi còn triển khai MLflow UI lên Hugging Face để theo dõi và quản lý các thí nghiệm và mô hình.

#### 12.6.1. Tạo Job deploy-mlflow-ui

Tôi đã thêm job thứ tư vào CI/CD pipeline để triển khai MLflow UI:

```yaml
deploy-mlflow-ui:
  runs-on: ubuntu-latest
  needs: train-model
  if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
  
  steps:
  - uses: actions/checkout@v4
  
  - name: Set up Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.10'
      
  - name: Download model artifacts
    uses: actions/download-artifact@v4
    with:
      name: model-artifacts
      path: downloaded_artifacts
      
  - name: Setup artifacts and mlruns
    run: |
      # Setup directories and copy artifacts
      mkdir -p models data mlruns
      
      # Copy model artifacts and restore mlruns data
      # ...
      
  - name: Prepare MLflow UI deployment
    run: |
      # Create deployment directory and copy templates
      # Set up environment and dependencies
      # Copy MLflow data to the deployment app
      # ...
  
  - name: Deploy MLflow UI to Hugging Face
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
      HF_USERNAME: ${{ secrets.HF_USERNAME }}
    run: |
      # Deploy the MLflow UI app to Hugging Face Space
      # ...
```

#### 12.6.2. Cách thức hoạt động của MLflow UI trên Hugging Face

MLflow UI được triển khai lên Hugging Face Spaces thông qua các bước chính:

1. **Sao lưu dữ liệu MLflow**: Trong quá trình huấn luyện, dữ liệu MLflow (experiments, runs, models) được sao lưu và lưu trữ dưới dạng artifact.

2. **Khôi phục dữ liệu**: Job deploy-mlflow-ui tải artifact và khôi phục dữ liệu MLflow.

3. **Chuẩn bị ứng dụng MLflow UI**:
   - Copy các template và file cấu hình
   - Đảm bảo các dependencies được cài đặt
   - Sao chép dữ liệu MLflow vào ứng dụng

4. **Fallback mechanism**: Nếu không có dữ liệu MLflow từ quá trình huấn luyện, script `setup_mlflow.py` sẽ tạo dữ liệu mẫu để đảm bảo MLflow UI không trống.

5. **Deploy lên Hugging Face**: Ứng dụng MLflow UI được đẩy lên Hugging Face Space sử dụng Git và Git LFS để quản lý các file nhị phân.

#### 12.6.3. Lợi ích của MLflow UI trên Hugging Face

Việc triển khai MLflow UI lên Hugging Face mang lại nhiều lợi ích:

1. **Truy cập từ xa**: Dễ dàng truy cập MLflow UI từ bất kỳ đâu mà không cần chạy server local.

2. **Chia sẻ**: Có thể chia sẻ kết quả thí nghiệm và thông tin mô hình với đồng nghiệp hoặc cộng đồng.

3. **Theo dõi**: Theo dõi quá trình huấn luyện và hiệu suất các mô hình theo thời gian thực.

4. **Tích hợp**: Tạo liên kết giữa ứng dụng dự đoán và thông tin mô hình để nâng cao tính minh bạch.

5. **Quản lý phiên bản**: Dễ dàng theo dõi và quản lý các phiên bản khác nhau của mô hình.

MLflow UI trên Hugging Face giúp hoàn thiện vòng đời MLOps bằng cách cung cấp một nền tảng quản lý và theo dõi thí nghiệm mà bất kỳ ai cũng có thể truy cập, tăng cường tính minh bạch và khả năng tái tạo trong quá trình phát triển mô hình. 