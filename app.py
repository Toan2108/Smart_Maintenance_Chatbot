import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Chatbot bảo trì", layout="wide")
st.title("🤖 AI Chatbot bảo trì từ Excel + OpenAI (Semantic Search)")

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df = df.dropna(how="all")
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

uploaded_file = st.file_uploader("📂 Tải lên Excel dữ liệu bảo trì (.xlsx)", type="xlsx")
if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"✅ Đã tải {len(df)} dòng dữ liệu")
    with st.expander("🔍 Xem trước dữ liệu"):
        st.dataframe(df, use_container_width=True)

    model = load_model()
    text_fields = df.select_dtypes(include="object").fillna("").astype(str)
    docs = text_fields.apply(lambda x: " | ".join(x), axis=1).tolist()
    doc_embeddings = model.encode(docs)

    query = st.text_input("💬 Nhập câu hỏi liên quan dữ liệu:")
    if query:
        query_embedding = model.encode([query])
        sims = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_idx = sims.argmax()
        top_score = sims[top_idx]

        st.markdown("### ✅ Kết quả gần nhất từ dữ liệu:")
        st.dataframe(df.iloc[[top_idx]])

        prompt = f"""Dữ liệu kỹ thuật: {docs[top_idx]}
Câu hỏi: {query}
Trả lời ngắn gọn, chính xác:"""

        if len(prompt) > 10000:
            st.warning("⚠️ Câu hỏi/dữ liệu vượt giới hạn token. Hệ thống đã rút gọn để tránh lỗi.")
            prompt = prompt[:10000]

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            st.success("🤖 Trả lời từ ChatGPT:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Lỗi: {e}")
