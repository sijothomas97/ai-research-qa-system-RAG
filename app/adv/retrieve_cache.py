from sentence_transformers import SentenceTransformer
import chromadb
import torch
from functools import lru_cache

# In-memory cache for query embeddings
@lru_cache(maxsize=100)
def encode_query(query):
    """
    Encode a query into an embedding, cached for reuse.
    Args:
        query (str): User query.
    Returns:
        list: Query embedding as a list.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    return model.encode([query])[0].tolist()

def retrieve_relevant_chunks(query, collection_name="arxiv_chunks", top_k=3):
    """
    Retrieve top-k relevant chunks from ChromaDB based on query.
    Args:
        query (str): User query.
        collection_name (str): ChromaDB collection name.
        top_k (int): Number of chunks to retrieve.
    Returns:
        list: List of relevant chunks with metadata.
    """
    # Generate query embedding (cached)
    query_embedding = encode_query(query)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)
    
    # Query the database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    retrieved_chunks = []
    for i in range(len(results['ids'][0])):
        retrieved_chunks.append({
            'chunk_id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'title': results['metadatas'][0][i]['title'],
            'paper_id': results['metadatas'][0][i]['paper_id'],
            'distance': results['distances'][0][i]
        })
    
    return retrieved_chunks

# Test retrieval
if __name__ == "__main__":
    query = "What are the latest advancements in neural networks?"
    results = retrieve_relevant_chunks(query)
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Chunk: {result['text']}")
        print(f"Distance: {result['distance']}\n")