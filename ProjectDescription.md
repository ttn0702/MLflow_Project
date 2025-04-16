# MÔ TẢ QUÁ TRÌNH THỰC HIỆN DỰ ÁN MLFLOW

## 1. Giới thiệu dự án

Dự án này nhằm tạo một mô hình phân loại đơn giản sử dụng MLflow để theo dõi quá trình thử nghiệm, lưu trữ mô hình và triển khai mô hình tốt nhất thông qua ứng dụng web Flask. Dự án đáp ứng các yêu cầu sau:

- Sử dụng dữ liệu tổng hợp từ hàm `make_classification` của scikit-learn
- Tạo mô hình phân loại đơn giản
- Thử nghiệm điều chỉnh siêu tham số
- So sánh kết quả của các mô hình
- Lưu trữ mô hình tốt nhất vào MLflow Model Registry
- Tạo ứng dụng web Flask để phục vụ các dự đoán từ mô hình tốt nhất

## 2. Cấu trúc dự án

```
MLflow_Project/
|-- data/              # Thư mục lưu trữ dữ liệu
|-- models/            # Thư mục lưu trữ mô hình đã huấn luyện
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

## 3. Các bước thực hiện

### Bước 1: Thiết lập môi trường và cài đặt thư viện

- Tạo thư mục dự án và cấu trúc thư mục
- Tạo file `requirements.txt` với các thư viện cần thiết:
  - scikit-learn: Cho việc tạo dữ liệu và xây dựng mô hình
  - pandas, numpy: Xử lý dữ liệu
  - mlflow: Theo dõi thí nghiệm và quản lý mô hình
  - flask: Xây dựng ứng dụng web
  - matplotlib: Trực quan hóa
  - joblib: Lưu và nạp mô hình

### Bước 2: Tạo dữ liệu phân loại

Trong file `train.py`, tôi đã triển khai hàm `generate_data()` để:
- Sử dụng `make_classification` từ scikit-learn để tạo dữ liệu phân loại nhị phân
- Chia dữ liệu thành tập huấn luyện và tập kiểm tra
- Chuẩn hóa dữ liệu bằng `StandardScaler`
- Lưu dữ liệu kiểm tra và bộ chuẩn hóa để sử dụng cho ứng dụng web sau này

### Bước 3: Thiết lập MLflow và huấn luyện mô hình

- Cấu hình MLflow tracking URI sử dụng thư mục cục bộ
- Tạo một MLflow experiment mới có tên "classification_experiment"
- Triển khai hàm `train_and_log_model()` để:
  - Huấn luyện một RandomForest classifier với các tham số khác nhau
  - Ghi lại các siêu tham số bằng `mlflow.log_param()`
  - Đánh giá mô hình trên tập kiểm tra
  - Ghi lại các chỉ số (accuracy, precision, recall, f1, roc_auc) bằng `mlflow.log_metric()`
  - Lưu mô hình với `mlflow.sklearn.log_model()`
  - Tạo và lưu biểu đồ feature importance

### Bước 4: Thử nghiệm điều chỉnh siêu tham số

- Định nghĩa một lưới các siêu tham số để thử nghiệm với RandomForest:
  - Số lượng cây (n_estimators): 100, 150, 200, 250
  - Độ sâu tối đa (max_depth): 10, 15, 20, None
  - Số lượng mẫu tối thiểu để phân tách (min_samples_split): 2, 5, 10
- Huấn luyện và đánh giá mô hình với mỗi bộ tham số
- Theo dõi mô hình có hiệu suất tốt nhất dựa trên điểm ROC AUC

### Bước 5: Đăng ký mô hình tốt nhất vào Model Registry

- Sau khi xác định mô hình tốt nhất, tiến hành lưu mô hình vào MLflow Model Registry
- Sử dụng `mlflow.sklearn.log_model()` với tham số `registered_model_name`
- Lưu mẫu mô hình tốt nhất cục bộ bằng joblib cho ứng dụng web

### Bước 6: Phát triển ứng dụng web Flask

- Tạo cấu trúc ứng dụng Flask trong thư mục `app/`
- Triển khai chức năng tải mô hình từ MLflow Model Registry hoặc từ tệp cục bộ
- Tạo hai endpoints:
  - `/`: Hiển thị form nhập liệu cho 20 features
  - `/predict`: Xử lý dữ liệu đầu vào, áp dụng bộ chuẩn hóa, và trả về dự đoán
- Thiết kế các template HTML sử dụng Bootstrap cho giao diện người dùng

### Bước 7: Tạo script trợ giúp để chạy dự án

- Triển khai file `run.py` để đơn giản hóa việc chạy các thành phần khác nhau:
  - `train`: Chạy quá trình huấn luyện mô hình
  - `ui`: Khởi chạy MLflow UI để khám phá các thí nghiệm
  - `app`: Khởi chạy ứng dụng web Flask
  - `all`: Chạy tất cả các thành phần theo trình tự

## 4. Kết quả và đánh giá

### Quá trình huấn luyện

Sau khi chạy quá trình huấn luyện, các mô hình với các bộ siêu tham số khác nhau được đánh giá. Mô hình tốt nhất đạt được:
- Accuracy: ~95%
- ROC AUC: ~0.98
- F1 Score: ~0.94

### MLflow UI

MLflow UI cho phép khám phá các thí nghiệm đã chạy, so sánh các mô hình, và trực quan hóa hiệu suất. Có thể truy cập qua http://localhost:5001.

### Ứng dụng web

Ứng dụng web cho phép người dùng:
- Nhập 20 giá trị đặc trưng (có thể ngẫu nhiên hóa)
- Nhận kết quả phân loại (0 hoặc 1)
- Xem xác suất dự đoán trực quan

## 5. Hướng dẫn chạy dự án

1. Cài đặt các phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

2. Huấn luyện mô hình:
   ```
   python run.py train
   ```

3. Khám phá kết quả với MLflow UI:
   ```
   python run.py ui
   ```

4. Chạy ứng dụng web:
   ```
   python run.py app
   ```

5. Chạy tất cả các bước:
   ```
   python run.py all
   ```

## 6. Kết luận

Dự án này đã thành công trong việc triển khai một quy trình MLOps đơn giản sử dụng MLflow, bao gồm:
- Quản lý thí nghiệm và theo dõi các siêu tham số
- So sánh và đánh giá các mô hình
- Lưu trữ và quản lý phiên bản mô hình
- Triển khai mô hình thông qua ứng dụng web

Thông qua dự án này, tôi đã học được cách sử dụng MLflow để theo dõi thí nghiệm, quản lý mô hình và triển khai mô hình vào ứng dụng thực tế. 