import pandas as pd
import pickle
import os
from utils import split_text, load_model, create_faiss_index

# Đọc dữ liệu Excel
df = pd.read_excel("data.xlsx")
text_fields = df.select_dtypes(include="object").fillna("").astype(str)
docs_raw = text_fields.apply(lambda x: " | ".join(x), axis=1).tolist()

# Chia nhỏ, mã hóa
chunks = []
source_map = []
for i, doc in enumerate(docs_raw):
    for chunk in split_text(doc):
        chunks.append(chunk)
        source_map.append(i)

model = load_model()
index, embeddings = create_faiss_index(chunks, model)

# Lưu lại FAISS index và dữ liệu
import faiss
faiss.write_index(index, "index.faiss")
with open("docs.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "source_map": source_map, "df": df}, f)

print("✅ Embedding & FAISS index created.")
