import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import pickle
import os
import torch

def setup_vector_db(chunks_file="chunked_papers.csv", collection_name="arxiv_chunks", cache_file="embeddings_cache.pkl"):
    """
    Generate or load cached embeddings and store in ChromaDB with HNSW tuning.
    Args:
        chunks_file (str): Path to chunked data CSV.
        collection_name (str): Name of ChromaDB collection.
        cache_file (str): Path to save/load cached embeddings.
    """
    # Load chunked data
    df = pd.read_csv(chunks_file)
    
    # Check for cached embeddings
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        # Initialize embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate embeddings
        embeddings = model.encode(df['chunk_text'].tolist(), show_progress_bar=True, batch_size=128)
        
        # Cache embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to {cache_file}")
    
    # Initialize ChromaDB client with HNSW configuration
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:M": 16, "hnsw:efConstruction": 200}  # Tune HNSW for speed
    )
    
    # Prepare data for ChromaDB
    documents = df['chunk_text'].tolist()
    ids = df['chunk_id'].tolist()
    metadatas = df[['paper_id', 'title']].to_dict('records')
    
    # Add to ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(documents)} chunks in ChromaDB collection '{collection_name}'")

# Run embedding and storage
if __name__ == "__main__":
    setup_vector_db()