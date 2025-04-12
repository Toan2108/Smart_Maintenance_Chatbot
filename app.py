# ✅ Gọi Streamlit config đầu tiên
import streamlit as st
# --- Khởi tạo bộ đếm truy cập phiên ---
if "visit_count" not in st.session_state:
    st.session_state.visit_count = 1
else:
    st.session_state.visit_count += 1

st.set_page_config(page_title="AI Chatbot Bảo Trì", layout="wide")

# --- Import thư viện ---
import openai
import os
import pickle
import faiss
import numpy as np
import requests
import zipfile
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import load_faiss_and_docs
import csv
from datetime import datetime
import socket

def log_visit(query_text):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        with open("visit_logs.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, ip, query_text])
    except Exception as e:
        st.warning(f"⚠️ Không thể ghi log truy cập: {e}")

import gdown

# --- Tùy chọn chế độ DEBUG ---
DEBUG = st.sidebar.checkbox("🛠 Hiện thông tin kiểm tra FAISS")
st.sidebar.markdown(f"🔢 **Lượt truy cập của bạn trong phiên này:** `{st.session_state.visit_count}`")
try:
    with open("visit_logs.csv", "r", encoding="utf-8") as f:
        total_visits = sum(1 for _ in f)
        st.sidebar.markdown(f"📈 **Tổng lượt truy cập:** {total_visits}")
except:
    st.sidebar.markdown("📈 **Tổng lượt truy cập:** 0")

# --- Hàm tải và giải nén mô hình từ Google Drive ---
def download_and_extract_model():
    file_id = "1R5j9GhJ-mHjxZh9HvIPRgPzZszWSedCM"
    zip_path = "local_model.zip"
    extract_folder = "local_model"

    if not os.path.exists(extract_folder):
        gdown.download(id=file_id, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_path)

    return extract_folder

# --- Load mô hình ---
model_path = download_and_extract_model()
model = SentenceTransformer(model_path)

# --- Load API key ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Tiêu đề giao diện ---
st.title("🤖 Smart Maintenance Chatbot")
st.markdown("Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu huấn luyện của Kỹ sư chuyên môn.")

# --- Load FAISS index và dữ liệu ---
index, docs = load_faiss_and_docs()

if isinstance(docs, dict):
    docs = list(docs.values())

# --- Nhập câu hỏi ---
query = st.text_input("💬 Nhập câu hỏi kỹ thuật hoặc lỗi máy móc:")

if query:
    log_visit(query)  # ✅ Ghi log khi có câu hỏi mới
    # Encode câu hỏi & tìm top-k
    query_embedding = model.encode([query])
    st.session_state.visit_count += 1  # Tăng lượt đếm khi người dùng đặt câu hỏi

    D, I = index.search(np.array(query_embedding), k=3)

    # ✅ DEBUG: Hiển thị kiểm tra nội bộ nếu được bật
    if DEBUG:
        st.write("📏 FAISS Distance (D):", D.tolist())
        st.write("🔢 FAISS Index (I):", I.tolist())
        st.write("📚 Tổng số đoạn văn (docs):", len(docs))
        st.write("🧾 Các đoạn dữ liệu tương ứng:")
        for i in I[0]:
            if 0 <= i < len(docs):
                st.code(docs[i])
            else:
                st.code(f"⛔ Không có dữ liệu phù hợp cho chỉ số {i}")

    # ✅ Chuẩn bị ngữ cảnh gửi cho OpenAI
    top_indices = I[0]
    contexts = [docs[i] for i in top_indices if 0 <= i < len(docs)]
    context = "\n\n".join(contexts) if contexts else "Không tìm thấy dữ liệu phù hợp."

    if not contexts:
        st.error("❌ Không tìm thấy đoạn dữ liệu phù hợp để trả lời.")
        st.stop()

    prompt = f"""
Bạn là chuyên gia kỹ thuật bảo trì. Dưới đây là dữ liệu liên quan:

--- Dữ liệu kỹ thuật ---
{context}

--- Câu hỏi ---
{query}

Vui lòng trả lời ngắn gọn, chính xác, dễ hiểu, và dựa vào thông tin từ DỮ LIỆU kỹ thuật bên trên và ChatGPT để đề xuất tối thiểu 3 giải pháp, bao gồm ít nhất 1 giải pháp phòng ngừa.
"""

    # ✅ Hiển thị các đoạn dữ liệu được dùng
    st.subheader("📄 Các đoạn dữ liệu được dùng:")
    for i, c in enumerate(contexts):
        st.markdown(f"**Đoạn {i+1}:**")
        st.code(c)

    # ✅ Gọi API OpenAI
    try:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()

        st.markdown("### 🤖 Kết quả từ AI:")
        st.success(answer)

        if DEBUG:
            with st.expander("📖 Dữ liệu chuẩn bị cho AI:"):
                st.code(context)

    except Exception as e:
        st.error(f"❌ Lỗi khi gọi OpenAI: {e}")
