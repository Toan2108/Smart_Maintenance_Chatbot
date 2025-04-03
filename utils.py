import os
import gdown

def download_if_not_exists(file_id: str, output_path: str):
    """
    Download a file from Google Drive if it does not exist locally.
    Args:
        file_id (str): The Google Drive file ID.
        output_path (str): The local path to save the file.
    """
    if not os.path.exists(output_path):
        print(f"🔽 Downloading {output_path} from Google Drive...")
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output=output_path, quiet=False)
        except Exception as e:
            print(f"❌ Failed to download {output_path}: {e}")
            raise

def load_faiss_and_docs():
    """
    Load or download FAISS index and docs.pkl from Google Drive.
    Returns:
        Tuple[str, str]: Paths to FAISS index and docs.pkl.
    """

    # Google Drive file IDs
    faiss_id = "1xw_y4wEHQSsdsTKDVUrDiKgJ-azKFzdB"
    docs_id = "1B9MXTHJHU98YCyDK03Dq9VdJuvMLSWDe"

    # Output local file paths
    faiss_path = "index.faiss"
    docs_path = "docs.pkl"

    # Download files if not exist
    download_if_not_exists(faiss_id, faiss_path)
    download_if_not_exists(docs_id, docs_path)

    return faiss_path, docs_path
