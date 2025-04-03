# 🤖 Smart Maintenance Chatbot (FAISS + GPT-3.5)

Ứng dụng AI hỗ trợ kỹ thuật bảo trì tìm kiếm thông tin từ Excel nội bộ, kết hợp semantic search và OpenAI GPT.

## ✅ Tính năng
- Load dữ liệu từ Excel 1 lần → sinh FAISS index
- Truy vấn cực nhanh không cần đọc lại file
- Kết hợp GPT-3.5 để trả lời thông minh

## 🚀 Hướng dẫn chạy local
```bash
pip install -r requirements.txt
python run_embedding.py  # tạo index từ data.xlsx
streamlit run app.py     # chạy chatbot
```

## ☁️ Deploy Streamlit Cloud
- Upload mã lên GitHub
- Tạo secret: `OPENAI_API_KEY`
- Deploy từ app.py

## 📁 File chính
- `app.py`: Chatbot giao diện
- `run_embedding.py`: Tạo FAISS từ Excel
- `utils.py`: Hàm xử lý text & embedding
- `docs.pkl`, `index.faiss`: dữ liệu học
