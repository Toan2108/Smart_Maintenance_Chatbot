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
import gdown

# --- Hàm tải và giải nén mô hình từ Google Drive ---
def download_and_extract_model():
    file_id = "https://drive.google.com/file/d/1DRWgv0tR2dBEWREsm3zgY5O2L6dlDm75/view?usp=sharing"
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

# --- Cấu hình Streamlit ---
st.set_page_config(page_title="AI Chatbot Bảo Trì", layout="wide")
st.title("🤖 Smart Maintenance Chatbot")
st.markdown("Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu huấn luyện nội bộ.")

# --- Load FAISS index và dữ liệu ---
faiss_path, docs_path = load_faiss_and_docs()
with open(docs_path, "rb") as f:
    docs = pickle.load(f)

index = faiss.read_index(faiss_path)

# --- Nhập câu hỏi ---
query = st.text_input("💬 Nhập câu hỏi kỹ thuật hoặc lỗi máy móc:")

if query:
    # Encode câu hỏi và tìm top-k văn bản
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    # Nếu docs là dict thì chuyển sang list
    if isinstance(docs, dict):
        docs = list(docs.values())

    # Lấy ngữ cảnh từ top-k đoạn văn
    top_indices = I[0]
    contexts = [docs[i] for i in top_indices if i != -1 and i < len(docs)]
    context = "\n\n".join(contexts) if contexts else "Không tìm thấy dữ liệu phù hợp."

    # Prompt cho OpenAI
    prompt = f"""
Bạn là chuyên gia kỹ thuật bảo trì. Dưới đây là dữ liệu liên quan:

--- Dữ liệu kỹ thuật ---
{context}

--- Câu hỏi ---
{query}

Vui lòng trả lời ngắn gọn, chính xác, và dễ hiểu.
"""

    # Hiển thị ngữ cảnh đã dùng
    st.subheader("📄 Các đoạn dữ liệu được dùng:")
    for i, c in enumerate(contexts):
        st.markdown(f"**Đoạn {i+1}:**")
        st.code(c)

    # Gọi API OpenAI
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
