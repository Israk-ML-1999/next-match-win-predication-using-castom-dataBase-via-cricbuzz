import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/new_match.json"
INDEX_PATH = "faiss_store/index.faiss"
EMBEDDINGS_PATH = "faiss_store/embeddings.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    texts = [json.dumps(entry) for entry in data]
    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    np.save(EMBEDDINGS_PATH, np.array(texts))

if __name__ == "__main__":
    try:
        build_faiss_index()
        print("FAISS index built successfully.")
    except Exception as e:
        print(f"Error: {e}")
