# ✅ Gọi Streamlit config đầu tiên
import streamlit as st
st.set_page_config(page_title="AI Chatbot Bảo Trì", layout="wide")
st.image("https://raw.githubusercontent.com/Toan2108/Smart_Maintenance_Chatbot/main/Logo.jpg", width=200)

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
import gdown

# --- Tùy chọn chế độ DEBUG ---
DEBUG = st.sidebar.checkbox("🛠 Hiện thông tin kiểm tra FAISS")

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
st.markdown("Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu huấn luyện nội bộ.")

# --- Load FAISS index và dữ liệu ---
faiss_path, docs_path = load_faiss_and_docs()
with open(docs_path, "rb") as f:
    docs = pickle.load(f)

if isinstance(docs, dict):
    docs = list(docs.values())

index = faiss.read_index(faiss_path)

# --- Nhập câu hỏi ---
query = st.text_input("💬 Nhập câu hỏi kỹ thuật hoặc lỗi máy móc:")

if query:
    # Encode câu hỏi & tìm top-k
    query_embedding = model.encode([query])
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

Vui lòng trả lời ngắn gọn, chính xác, dễ hiểu, và dựa vào thông tin từ DỮ LIỆU NỘI BỘ bên trên và ChatGPT để đề xuất tối thiểu 3 giải pháp.
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
