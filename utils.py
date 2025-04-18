MODEL_FILE_ID = "1zVhluWQbAQ0eNdRLxflTdqssilc9E2cz"
MODEL_PATH = "/tmp/local_model"
import zipfile

def download_and_extract_model():
    zip_path = "/tmp/all-mpnet-base-v2.zip"
    download_file(MODEL_FILE_ID, zip_path)

    # Giải nén
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_PATH)

    return MODEL_PATH

def load_faiss_and_docs(index_path="/tmp/index.faiss", docs_path="/tmp/docs.pkl"):
    import faiss
    import pickle

    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    return index, docs
import pickle
import faiss
import os
import gdown

# ✅ ID file Google Drive (public link)
DOCS_ID = "1a9gjf3uSoU14LrzydfPaksMBDvjArINS"
INDEX_ID = "15sT4YLUUozCax9Sy8rHqC91vTU9toL0w"

DOCS_PATH = "/tmp/docs.pkl"
INDEX_PATH = "/tmp/index.faiss"

def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"✅ Đã có sẵn: {output_path}")

def load_faiss_and_docs():
    download_file(DOCS_ID, DOCS_PATH)
    download_file(INDEX_ID, INDEX_PATH)

    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    print("✅ Đã load xong FAISS và dữ liệu.")
    return index, docs
