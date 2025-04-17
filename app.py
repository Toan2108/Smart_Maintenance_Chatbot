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
Bạn là một kỹ sư bảo trì chuyên nghiệp. Dưới đây là một số thông tin kỹ thuật liên quan đã được cung cấp từ tài liệu nội bộ:

{context_text}

Dựa trên tài liệu kỹ thuật ở trên, hãy phân tích và đưa ra câu trả lời chính xác cho câu hỏi sau:

Câu hỏi: {query}

Yêu cầu:
- Đưa ra ít nhất **03 giải pháp cụ thể** để xử lý vấn đề.
- Giải pháp phải **liên quan đến nội dung câu hỏi và thông tin kỹ thuật được cung cấp ở trên**.
- Thêm **01 giải pháp mang tính phòng ngừa** để tránh sự cố tái diễn.
- Thêm **01 giải pháp từ ChatGPT, lưu ý: chỉ thêm thông tin ngoài tài liệu nếu thông tin chắc chắn**.
- Câu trả lời cần **ngắn gọn, rõ ràng, dễ hiểu**, viết dưới dạng **liệt kê đánh số** (1, 2, 3) cho 3 giải pháp đầu tiên, các giải pháp tiếp theo không đánh số.

Chỉ trả lời đúng trọng tâm, không lặp lại câu hỏi.
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
