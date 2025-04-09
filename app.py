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

# --- Cấu hình Streamlit ---
# --- Tùy chọn bật chế độ kiểm tra FAISS ---
DEBUG = st.sidebar.checkbox("🛠 Hiện thông tin kiểm tra FAISS")
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
# ✅ In khoảng cách và chỉ số để kiểm tra FAISS
# ✅ Chuyển docs về list nếu là dict
if isinstance(docs, dict):
    docs = list(docs.values())

# ✅ DEBUG: Kiểm tra chỉ số và độ dài
if DEBUG:
    st.write("📏 FAISS Distance (D):", D.tolist())
    st.write("🔢 FAISS Index (I):", I.tolist())
    st.write("📚 Tổng số đoạn văn (docs):", len(docs))

st.write("🔍 Khoảng cách FAISS trả về (D):", D.tolist())
st.write("🔢 Chỉ số FAISS trả về (I):", I.tolist())

# Nếu muốn kiểm tra nội dung từng đoạn:
if DEBUG:
    st.write("🧾 Các đoạn dữ liệu tương ứng:")
    for i in I[0]:
        if 0 <= i < len(docs):
            st.code(docs[i])
        else:
            st.code(f"⛔ Không có dữ liệu phù hợp cho chỉ số {i}")

    # Nếu docs là dict thì chuyển sang list
    if isinstance(docs, dict):
        docs = list(docs.values())

# ✅ Lọc các đoạn văn bản hợp lệ từ chỉ số FAISS
top_indices = I[0]
contexts = []

for idx in top_indices:
    if 0 <= idx < len(docs):
        contexts.append(docs[idx])
    else:
        st.warning(f"⚠️ Chỉ số {idx} vượt ngoài phạm vi docs.")

    context = "\n\n".join(contexts) if contexts else "Không tìm thấy dữ liệu phù hợp."
if not contexts:
    st.error("❌ Không tìm thấy đoạn dữ liệu phù hợp để trả lời.")
    st.stop()

    # Prompt cho OpenAI
    prompt = f"""
Bạn là chuyên gia kỹ thuật bảo trì. Dưới đây là dữ liệu liên quan:

--- Dữ liệu kỹ thuật ---
{context}

--- Câu hỏi ---
{query}

Vui lòng trả lời ngắn gọn, chính xác, và dễ hiểu, và chỉ dựa vào thông tin từ DỮ LIỆU NỘI BỘ bên trên.
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
