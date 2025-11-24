from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
import torch
from functools import lru_cache

# In-memory cache for query embeddings
@lru_cache(maxsize=100)
def encode_query(query):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    return model.encode([query])[0].tolist()

def retrieve_relevant_chunks(query, collection_name="arxiv_chunks", top_k=5, hybrid_weight=0.5):
    """
    Retrieve top-k relevant chunks using hybrid vector + BM25 search and re-ranking.
    Args:
        query (str): User query.
        collection_name (str): ChromaDB collection name.
        top_k (int): Number of chunks to retrieve.
        hybrid_weight (float): Weight for vector search (0.0 = BM25 only, 1.0 = vector only).
    Returns:
        list: List of relevant chunks with metadata.
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)
    
    # Get all documents for BM25
    all_docs = collection.get(include=['documents', 'metadatas', 'ids'])['documents']
    bm25 = BM25Okapi([doc.split() for doc in all_docs])
    bm25_scores = bm25.get_scores(query.split())
    
    # Vector search
    query_embedding = encode_query(query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2  # Retrieve more for re-ranking
    )
    
    # Combine scores
    vector_scores = {id: 1 - dist for id, dist in zip(vector_results['ids'][0], vector_results['distances'][0])}
    combined_results = []
    for i, id in enumerate(vector_results['ids'][0]):
        bm25_score = bm25_scores[all_docs.index(vector_results['documents'][0][i])]
        combined_score = hybrid_weight * vector_scores[id] + (1 - hybrid_weight) * bm25_score
        combined_results.append({
            'chunk_id': id,
            'text': vector_results['documents'][0][i],
            'title': vector_results['metadatas'][0][i]['title'],
            'paper_id': vector_results['metadatas'][0][i]['paper_id'],
            'score': combined_score
        })
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Re-rank with cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    pairs = [(query, result['text']) for result in combined_results[:top_k * 2]]
    cross_scores = cross_encoder.predict(pairs)
    
    # Combine cross-encoder scores
    for i, result in enumerate(combined_results[:top_k * 2]):
        result['score'] = 0.7 * cross_scores[i] + 0.3 * result['score']
    
    # Final sorting and trimming
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    return combined_results[:top_k]

# Test retrieval
if __name__ == "__main__":
    query = "What are the latest advancements in neural networks?"
    results = retrieve_relevant_chunks(query)
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Chunk: {result['text']}")
        print(f"Score: {result['score']}\n")