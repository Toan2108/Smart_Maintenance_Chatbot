import streamlit as st
import openai
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ✅ [MỚI] Dùng normalize + cosine matching tương thích FAISS IndexFlatIP
# ✅ [MỚI] Dùng model all-mpnet-base-v2 để tăng độ chính xác semantic

# --- Cấu hình trang ---
st.set_page_config(page_title="AI Chatbot Bảo Trì", layout="wide")
st.title("🤖 Smart Maintenance Chatbot")
st.markdown("Chatbot hỗ trợ kỹ thuật viên tra cứu lỗi & hướng xử lý từ dữ liệu kỹ thuật nội bộ.")

# --- Load API Key từ biến môi trường hoặc secrets ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load FAISS index và văn bản gốc ---
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
if isinstance(docs, dict):
    docs = list(docs.values())

index = faiss.read_index("index.faiss")

# --- Nhập câu hỏi từ người dùng ---
query = st.text_input("💬 Nhập câu hỏi kỹ thuật hoặc lỗi máy móc:")

if query:
    # ✅ [CẬP NHẬT] Dùng model embedding mạnh hơn
    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding, axis=1)

    # ✅ [CẬP NHẬT] FAISS IndexFlatIP tương thích cosine
    D, I = index.search(np.array(query_embedding), k=3)

    # --- Hiển thị ngữ cảnh tìm được ---
    st.subheader("📄 Dữ liệu tham chiếu:")
    contexts = []
    for i in I[0]:
        if i >= 0 and i < len(docs):
            st.markdown(f"- {docs[i]}")
            contexts.append(docs[i])

    context = "\n\n".join(contexts)

    # --- Gửi prompt tới GPT ---
    prompt = f"""
Bạn là chuyên gia kỹ thuật bảo trì. Dưới đây là dữ liệu kỹ thuật nội bộ:

{context}

--- Câu hỏi ---
{query}

Vui lòng trả lời chính xác, ngắn gọn và dễ hiểu.
"""

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        st.markdown("### 🤖 Trả lời từ AI:")
        st.success(answer)
    except Exception as e:
        st.error(f"Lỗi khi gọi OpenAI API: {e}")
