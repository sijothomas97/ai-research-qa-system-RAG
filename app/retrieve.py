# File: retrieve.py
from sentence_transformers import SentenceTransformer
import chromadb

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
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
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