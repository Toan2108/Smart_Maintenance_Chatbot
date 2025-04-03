import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def split_text(text, max_length=300):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

def create_faiss_index(docs, model):
    embeddings = model.encode(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings
