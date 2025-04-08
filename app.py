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
    D, I = index.search(np.array(query_embedding), k=3)
# Nếu docs là dict thì chuyển sang list
if isinstance(docs, dict):
    docs = list(docs.values())

# Lấy nhiều đoạn context từ chỉ số trả về (k=3)
top_indices = I[0]
contexts = []

for idx in top_indices:
    if idx != -1 and idx < len(docs):
        contexts.append(docs[idx])

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
st.subheader("🧾 Các đoạn dữ liệu được dùng:")
for i, c in enumerate(contexts):
    st.markdown(f"**Đoạn {i+1}:**")
    st.code(c)

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

