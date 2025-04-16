
# 🤖 Smart Maintenance Chatbot

Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu huấn luyện nội bộ của kỹ sư chuyên môn. Ứng dụng công nghệ **FAISS** và **mô hình ngữ nghĩa all-mpnet-base-v2** để tìm kiếm ngữ cảnh chính xác nhất trong dữ liệu kỹ thuật được đào tạo trước.

---

## ✅ Tính năng

- Truy vấn tự nhiên bằng tiếng Việt cho các lỗi máy móc kỹ thuật.
- Tìm kiếm câu trả lời liên quan nhất trong dữ liệu được huấn luyện nội bộ.
- Hiển thị **các đoạn dữ liệu tham chiếu** liên quan (top 3).
- Phân tích và hiển thị **gợi ý AI** dựa trên dữ liệu kỹ thuật.
- Giao diện Streamlit trực quan, dễ sử dụng.

---

## ⚙️ Công nghệ sử dụng

| Thành phần | Mô tả |
|------------|------|
| **FAISS** | Để tạo và tìm kiếm index vector |
| **SentenceTransformer (mpnet)** | Dùng mô hình `all-mpnet-base-v2` để tạo embedding |
| **Streamlit** | Giao diện web đơn giản, dễ deploy |
| **Google Drive** | Lưu trữ mô hình nén `.zip`, docs và index |
| **Torch + Transformers** | Backend model xử lý dữ liệu |

---

## 🧱 Cấu trúc file quan trọng

```
Smart_Maintenance_Chatbot/
├── app.py                      # Ứng dụng Streamlit chính
├── utils.py                   # Hàm tiện ích tải mô hình từ Google Drive và load dữ liệu FAISS
├── requirements.txt           # Danh sách thư viện cần cài
├── .env.example               # File mẫu khai báo API Key
├── index.faiss                # FAISS Index vector (tải từ Drive)
├── docs.pkl                   # Danh sách câu hỏi – đáp kỹ thuật (tải từ Drive)
└── all-mpnet-base-v2.zip      # Mô hình embedding đã nén (tải từ Drive)
```

---

## ☁️ Hướng dẫn Deploy trên [Streamlit Cloud](https://streamlit.io/cloud)

1. **Fork hoặc clone repo**
2. **Đặt các file sau lên Google Drive** (ở chế độ Public or "Anyone with link"):
   - `all-mpnet-base-v2.zip`
   - `docs.pkl`
   - `index.faiss`
3. **Cập nhật `utils.py`**
   - Chứa các hàm `download_file()` để tự động tải và giải nén mô hình từ Google Drive.
4. **Cập nhật file `.env` hoặc nhập `OPENAI_API_KEY` vào phần `Secrets`**
5. **Deploy app** và truy cập URL công khai.

---

## 🔐 Secrets cần thiết

Trong phần **App Settings → Secrets**, nhập như sau:

```toml
OPENAI_API_KEY = "sk-..."
```

---

## ✨ Ví dụ truy vấn

> **Nhập:** “Robot không về home”  
> **Kết quả:** Chatbot trả về 3 đoạn dữ liệu kỹ thuật từ file huấn luyện liên quan đến robot không về Home, gợi ý nguyên nhân & hướng xử lý chi tiết.

---

## 📦 Yêu cầu cài đặt (nếu chạy local)

```bash
pip install -r requirements.txt
```

---

## ☎️ Liên hệ & bản quyền

- Tác giả: **Toan Nguyen**
- Dành riêng cho môi trường công nghiệp & đội ngũ kỹ thuật viên nhà máy.
- Sử dụng dữ liệu được sàng lọc bởi Kỹ sư chuyên môn.
