# MLflow Project

![MLflow Project CI/CD](https://github.com/[your-username]/MLflow_Project/workflows/MLflow%20Project%20CI/CD/badge.svg)

## Giới thiệu

Dự án này triển khai một quy trình MLOps hoàn chỉnh sử dụng MLflow, từ việc tạo dữ liệu tổng hợp, huấn luyện mô hình phân loại nhị phân, đánh giá và lưu trữ mô hình với MLflow, đến triển khai mô hình thông qua ứng dụng web Flask.

## Tính năng

- Tạo dữ liệu tổng hợp cho bài toán phân loại nhị phân
- Huấn luyện mô hình RandomForest với điều chỉnh siêu tham số
- Ghi lại thí nghiệm, siêu tham số và chỉ số đánh giá với MLflow
- Trực quan hóa độ quan trọng của đặc trưng
- Lưu trữ mô hình tốt nhất vào MLflow Model Registry
- Triển khai mô hình thông qua ứng dụng web Flask
- CI/CD pipeline với GitHub Actions

## Cấu trúc dự án

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
|-- tests/             # Unit tests
|-- .github/workflows/ # GitHub Actions CI/CD
|-- train.py           # Script chính để tạo dữ liệu và huấn luyện mô hình
|-- run.py             # Script hỗ trợ để chạy các thành phần khác nhau
|-- requirements.txt   # Các gói phụ thuộc
|-- README.md          # Tệp README
```

## Cài đặt

```bash
# Clone repository
git clone https://github.com/[your-username]/MLflow_Project.git
cd MLflow_Project

# Cài đặt dependencies
pip install -r requirements.txt
```

## Sử dụng

Huấn luyện mô hình:
```bash
python run.py train
```

Khám phá kết quả với MLflow UI:
```bash
python run.py ui
```

Chạy ứng dụng web:
```bash
python run.py app
```

Chạy tất cả các thành phần:
```bash
python run.py all
```

## CI/CD với GitHub Actions

Dự án sử dụng GitHub Actions để tự động hóa các quy trình CI/CD:

1. **Kiểm tra linting và unit test**: Chạy kiểm tra code style với flake8 và kiểm tra đơn vị với pytest
2. **Huấn luyện mô hình**: Tự động huấn luyện mô hình khi code được đẩy lên nhánh main/master
3. **Triển khai ứng dụng**: Tự động triển khai ứng dụng web sau khi huấn luyện mô hình

Xem file `./github/workflows/mlflow-ci.yml` để biết chi tiết về cấu hình CI/CD.

## Đóng góp

Vui lòng đóng góp bằng cách tạo issue hoặc pull request. Mọi đóng góp đều được hoan nghênh!

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết. 