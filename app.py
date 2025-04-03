import streamlit as st
import openai
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import load_faiss_and_docs

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
    # Encode câu hỏi
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    # Tìm văn bản gần nhất
    D, I = index.search(np.array(query_embedding), k=1)
top_idx = I[0][0]

# Nếu docs là dict → chuyển sang list
if isinstance(docs, dict):
    docs = list(docs.values())

# Xử lý lỗi nếu chỉ số vượt quá độ dài
if top_idx < len(docs):
    context = docs[top_idx]
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

    # Gọi API GPT-3.5
# Gọi API GPT-3.5
try:
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    # Hiển thị kết quả
    st.markdown("### 🤖 Kết quả từ AI:")
    st.success(answer)

    with st.expander("📖 Dữ liệu chuẩn bị cho AI:"):
        st.code(context)

except Exception as e:
    st.error(f"❌ Lỗi khi gọi OpenAI: {e}")

