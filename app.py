import streamlit as st
import openai
import os
os.environ["TORCH_DISABLE_WATCHDOG"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/tmp"  # nơi lưu cache model
os.environ["HF_DATASETS_CACHE"] = "/tmp"
os.environ["HF_METRICS_CACHE"] = "/tmp"

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

from sklearn.preprocessing import normalize
from utils import load_faiss_and_docs

# ✅ Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Cấu hình giao diện
st.set_page_config(page_title="Smart Maintenance Chatbot", layout="wide")
st.title("🤖 Smart Maintenance Chatbot")
st.markdown("Nhập câu hỏi kỹ thuật để được hỗ trợ từ Kỹ sư chuyên môn.")

# ✅ Load dữ liệu FAISS và văn bản
index, docs = load_faiss_and_docs()

# ✅ Load mô hình embedding
from utils import download_and_extract_model

# Load mô hình từ local (đã được giải nén)
model_path = download_and_extract_model()
model = SentenceTransformer(model_path)

# ✅ Nhập câu hỏi từ người dùng
query = st.text_input("🛠️ Nhập câu hỏi kỹ thuật:")

if query:
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding, axis=1)

    D, I = index.search(query_embedding, k=3)

    contexts = []
    for idx in I[0]:
        if 0 <= idx < len(docs):
            contexts.append(docs[idx])

    if not contexts:
        st.error("Không tìm thấy thông tin phù hợp.")
    else:
        st.subheader("📎 Tài liệu tham chiếu:")
        for i, ctx in enumerate(contexts):
            st.markdown(f"**{i+1}.** {ctx}")
        context_text = "\n\n".join(contexts)
        prompt = f"""
Bạn là chuyên gia kỹ thuật, kỹ sư Bảo trì. Dưới đây là một số thông tin kỹ thuật liên quan:

{context_text}

Câu hỏi: {query}
Vui lòng trả lời chính xác, rõ ràng, ngắn gọn. Dựa vào dữ liệu kỹ thuật ở trên, đề xuất ít nhất 3 giải pháp và có 1 giải pháp phòng ngừa, các đề xuất phải liên quan đến nội dung của câu hỏi.
"""

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
        except Exception as e:
            st.error(f"Lỗi khi gọi OpenAI API: {e}")
