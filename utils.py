import os
import pickle
import faiss
import gdown

# ✅ Nhập ID Google Drive tại đây
DOCS_FILE_ID = "1BqEbPJBb_GTq9b0qhivSD038zOGh2juJ"       # docs.pkl
INDEX_FILE_ID = "1R5iSgWbdlmCMwB9SKJw8yEBOngR9cfhj"      # index.faiss

# ✅ Đường dẫn tạm để lưu khi tải file từ Google Drive
DOCS_PATH = "/tmp/docs.pkl"
INDEX_PATH = "/tmp/index.faiss"

# -------------------------------
# 📥 Hàm tải file từ Google Drive
# -------------------------------
def download_from_google_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        print(f"📥 Đang tải {output_path} từ Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"✅ Đã có file {output_path}, bỏ qua tải lại.")

# --------------------------------
# 🔄 Hàm load docs và FAISS index
# --------------------------------
def load_faiss_and_docs():
    # B1: Tải file nếu chưa có
    download_from_google_drive(DOCS_FILE_ID, DOCS_PATH)
    download_from_google_drive(INDEX_FILE_ID, INDEX_PATH)

    # B2: Load FAISS index
    print("📦 Đang load FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    # B3: Load docs
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    print("✅ Load thành công FAISS index và dữ liệu.")
    return index, docs
