import streamlit as st
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
import gdown  # thêm thư viện này ở đầu

# Hàm tải file zip từ Google Drive
def download_and_extract_model():
    model_url = "https://drive.google.com/uc?id=1GwQQmdZ2O2wGixLKiRk9MiMtBfToozll"  # đúng định dạng gdown
    zip_path = "local_model.zip"
    extract_folder = "local_model"

    if not os.path.exists(extract_folder):
        # Tải file zip bằng gdown
        gdown.download(model_url, zip_path, quiet=False)

        # Giải nén file zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        os.remove(zip_path)  # Xóa file zip sau khi giải nén

    return extract_folder

# Load API Key từ .env hoặc secrets
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cấu hình Streamlit
st.set_page_config(page_title="AI Chatbot Bảo Trì", layout="wide")
st.title("🤖 Smart Maintenance Chatbot")
st.markdown("Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu huấn luyện nội bộ.")

# Bước 1: Tải dữ liệu từ Google Drive nếu chưa có
faiss_path, docs_path = load_faiss_and_docs()

# Bước 2: Load FAISS index và dữ liệu gốc
with open(docs_path, "rb") as f:
    docs = pickle.load(f)
index = faiss.read_index(faiss_path)

# Bước 3: Nhận câu hỏi từ người dùng
query = st.text_input("💬 Nhập câu hỏi kỹ thuật hoặc lỗi máy móc:")

if query:
    # Tải model local nếu chưa có
    model_path = download_and_extract_model()
    model = SentenceTransformer(model_path)
    query_embedding = model.encode([query])

    # FAISS tìm top 3 đoạn gần nhất
    D, I = index.search(np.array(query_embedding), k=3)

    # Xử lý dữ liệu nếu là dict
    if isinstance(docs, dict):
        docs = list(docs.values())

    # Lấy nhiều đoạn context
    top_indices = I[0]
    contexts = [docs[idx] for idx in top_indices if idx != -1 and idx < len(docs)]

    if contexts:
        context = "\n\n".join(contexts)
    else:
        context = "Không tìm thấy dữ liệu phù hợp."

    # Tạo prompt cho OpenAI
    prompt = f"""
Bạn là chuyên gia kỹ thuật bảo trì. Dưới đây là dữ liệu liên quan:

--- Dữ liệu kỹ thuật ---
{context}

--- Câu hỏi ---
{query}

Vui lòng trả lời ngắn gọn, chính xác, và dễ hiểu.
"""

    # Hiển thị dữ liệu dùng
    st.subheader("🧾 Các đoạn dữ liệu được dùng:")
    for i, c in enumerate(contexts):
        st.markdown(f"**Đoạn {i+1}:**")
        st.code(c)

    # Gọi OpenAI GPT-3.5
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

        with st.expander("📖 Dữ liệu chuẩn bị cho AI:"):
            st.code(context)

    except Exception as e:
        st.error(f"❌ Lỗi khi gọi OpenAI: {e}")
