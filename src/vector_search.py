import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dynamically determine the embedding dimension from the model
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)  # Initialize FAISS index with correct dimension

# Example cricket match data for vector database (in real use, you'd load this from a file)
match_data = [
    "India vs Australia - India will win the next match.",
    "Australia is likely to win their next match against India.",
    "The next cricket match between India and Australia will be intense.",
]

# Convert match data to embeddings
embeddings = model.encode(match_data)

# Add embeddings to FAISS index
index.add(np.array(embeddings))

def search_vector_database(query):
    """Search the FAISS index for the most similar cricket match prediction."""
    print(f"🔍 Searching for: {query}")
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=3)
    
    # If no match found (in case of bad query or empty index)
    if len(indices) == 0:
        return []
    
    # Return results based on indices
    results = []
    for i in indices[0]:
        if i >= 0:  # If valid index
            results.append(match_data[i])
    
    return results
