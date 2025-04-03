import os
import gdown

def download_if_not_exists(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        gdown.download(id=file_id, output=output_path, quiet=False)

def load_faiss_and_docs():
    # Google Drive file IDs
    faiss_id = "1xw_y4wEHQSsdsTKDVUrDiKgJ-azKFzdB"
    docs_id = "1B9MXTHJHU98YCyDK03Dq9VdJuvMLSWDe"

    # File paths
    faiss_path = "index.faiss"
    docs_path = "docs.pkl"

    # Download if needed
    download_if_not_exists(faiss_id, faiss_path)
    download_if_not_exists(docs_id, docs_path)

    return faiss_path, docs_path

