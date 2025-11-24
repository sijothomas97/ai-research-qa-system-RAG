# File: embed_and_store.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

def setup_vector_db(chunks_file="../data/chunked_papers.csv", collection_name="arxiv_chunks"):
    """
    Generate embeddings for chunks and store in ChromaDB.
    Args:
        chunks_file (str): Path to chunked data CSV.
        collection_name (str): Name of ChromaDB collection.
    """
    # Load chunked data
    df = pd.read_csv(chunks_file)
    
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2') # is a lightweight, efficient model designed for general-purpose sentence embeddings.
    
    # Generate embeddings
    embeddings = model.encode(df['chunk_text'].tolist(), show_progress_bar=True) # convert_to_tensor: True (Returns Pytorch tensors for GPU processing)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collection
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)
    
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
setup_vector_db()