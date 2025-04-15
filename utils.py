import faiss
import pickle
import os
import gdown

# ✅ ID file Google Drive (cập nhật theo file thật của bạn)
DOCS_ID = "1B9MXTHJHU98YCyDK03Dq9VdJuvMLSWDe"       # docs.pkl
INDEX_ID = "1xw_y4wEHQSsdsTKDVUrDiKgJ-azKFzdB"      # index.faiss

def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"✅ Đã có sẵn: {output_path}")

def load_faiss_and_docs():
    download_file(DOCS_ID, "docs.pkl")
    download_file(INDEX_ID, "index.faiss")

    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

    index = faiss.read_index("index.faiss")
    return index, docs
