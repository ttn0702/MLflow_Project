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
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import sys
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Tải mô hình từ MLflow hoặc tệp cục bộ
def load_model():
    try:
        # Tải từ MLflow model registry
        mlflow.set_tracking_uri("file:../mlruns")
        model = mlflow.sklearn.load_model("models:/BestClassificationModel/latest")
        print("Loaded model from MLflow model registry")
    except Exception as e:
        print(f"Failed to load model from MLflow registry: {e}")
        # Sử dụng mô hình cục bộ nếu thất bại
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib")
        model = joblib.load(model_path)
        print("Loaded model from local file")
    
    return model

# Tải bộ chuẩn hóa
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scaler.joblib")
    return joblib.load(scaler_path)

# Tải mô hình và bộ chuẩn hóa khi khởi động
model = load_model()
scaler = load_scaler()
```

### 7.2. Tạo API endpoint

Tôi đã tạo hai endpoints chính:

```python
@app.route('/')
def home():
    return render_template('index.html')

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
            features = np.array(features).reshape(1, -1)
        
        # Chuẩn hóa đặc trưng
        scaled_features = scaler.transform(features)
        
        # Dự đoán
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability)
        }
        
        # Nếu yêu cầu từ form, trả về template
        if not request.is_json:
            return render_template('result.html', prediction=result['prediction'], 
                                  probability=round(result['probability'] * 100, 2))
        
        # Ngược lại trả về JSON
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
```

### 7.3. Phát triển giao diện người dùng

Tôi đã thiết kế hai template HTML cho ứng dụng:

1. **index.html** - Form nhập liệu:
   - Hiển thị form với 20 trường đầu vào cho các đặc trưng
   - Nút ngẫu nhiên hóa để tạo giá trị ngẫu nhiên
   - Hiện thị hướng dẫn và thông tin về ứng dụng

2. **result.html** - Trang kết quả:
   - Hiển thị kết quả phân loại (Lớp 0 hoặc Lớp 1)
   - Thanh tiến trình trực quan hiển thị xác suất dự đoán
   - Nút để quay lại và thực hiện dự đoán mới

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