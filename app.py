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

def download_and_extract_model():
    # Link Google Drive dạng ID (đã có quyền chia sẻ công khai)
    file_id = "1GwQQmdZ2O2wGixLKiRk9MiMtBfToozll"
    zip_path = "local_model.zip"
    extract_folder = "local_model"

    if not os.path.exists(extract_folder):
        gdown.download(id=file_id, output=zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_path)

    return extract_folder

# --- Tải mô hình từ Google Drive ---
def download_and_extract_model():
    model_url = "https://drive.google.com/uc?id=1GwQQmdZ2O2wGixLKiRk9MiMtBfToozll"
    zip_path = "local_model.zip"
    extract_folder = "local_model"

    if not os.path.exists(extract_folder):
        with requests.get(model_url, stream=True) as r:
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_path)

    return extract_folder

# --- Tải model ---
model_path = download_and_extract_model()
model = SentenceTransformer(model_path)

# --- Load API key ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Cấu hình Streamlit ---
st.set_page_config(page_title="AI Chatbot Bảo Trì", layout="wide")
st.title("🤖 Smart Maintenance Chatbot")
st.markdown("Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu huấn luyện nội bộ.")

# --- Load FAISS index và docs ---
faiss_path, docs_path = load_faiss_and_docs()
with open(docs_path, "rb") as f:
    docs = pickle.load(f)
index = faiss.read_index(faiss_path)

# --- Nhận câu hỏi ---
query = st.text_input("💬 Nhập câu hỏi kỹ thuật hoặc lỗi máy móc:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    if isinstance(docs, dict):
        docs = list(docs.values())

    top_indices = I[0]
    contexts = [docs[i] for i in top_indices if i != -1 and i < len(docs)]

    if contexts:
        context = "\n\n".join(contexts)
    else:
        context = "Không tìm thấy dữ liệu phù hợp."

    prompt = f"""
Bạn là chuyên gia kỹ thuật bảo trì. Dưới đây là dữ liệu liên quan:

--- Dữ liệu kỹ thuật ---
{context}

--- Câu hỏi ---
{query}

Vui lòng trả lời ngắn gọn, chính xác, và dễ hiểu.
"""

    st.subheader("🧾 Các đoạn dữ liệu được dùng:")
    for i, c in enumerate(contexts):
        st.markdown(f"**Đoạn {i+1}:**")
        st.code(c)

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
