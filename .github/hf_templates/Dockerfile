FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . .

# Mở cổng 7860 - cổng mặc định của Hugging Face Spaces
EXPOSE 7860

# Chạy ứng dụng với gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"] 