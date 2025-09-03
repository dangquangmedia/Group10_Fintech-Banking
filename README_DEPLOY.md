# Đồ án: Đánh giá điểm tín dụng cá nhân trong Fintech & Banking

## Cấu trúc
```
credit-scoring-project/
├── app.py                  # Streamlit app
├── requirements.txt
├── README_DEPLOY.md
├── data/
│   ├── raw/                # Dữ liệu gốc (đặt credit_data.csv tại đây)
│   └── processed/          # Dữ liệu sau tiền xử lý
├── models/                 # Mô hình đã train (.pkl)
├── reports/                # Hình biểu đồ & architecture
└── src/
    ├── data_preprocessing.py
    ├── train_model.py
    ├── evaluate.py
    └── utils.py
```

## Chạy local
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Cloud
1. Push toàn bộ thư mục này lên GitHub.
2. Vào https://share.streamlit.io/ → New app → trỏ tới repo + branch.
3. **Main file**: `app.py`
4. **Python version**: 3.10/3.11, add secrets nếu dùng API ngoài.

## Deploy Google Cloud Run (tùy chọn)
Tạo `Dockerfile`:
```
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build & deploy:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/credit-scoring
gcloud run deploy credit-scoring --image gcr.io/PROJECT_ID/credit-scoring --platform managed --allow-unauthenticated
```

## Ghi chú
- Nếu **chưa có** file mô hình trong `models/`, app vẫn chạy với **chế độ mô phỏng** để demo.
- Đặt dữ liệu Kaggle tại `data/raw/credit_data.csv` để train lại.
