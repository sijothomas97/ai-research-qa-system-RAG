from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize model and vectorstore
model = SentenceTransformer("all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(
    ["Phone X: 12-hour battery life", "Phone X: 128GB storage", ...],
    model.encode,
    metadatas=[{"id": 0}, {"id": 1}, ...]
)

# Validation set: query, relevant chunk ID
val_data = [
    {"query": "What’s the battery life of Phone X?", "relevant_id": 0},
    # ... more samples
]

# Evaluate retrieval
precision_scores = []
cosine_scores = []
for sample in val_data:
    query_embedding = model.encode([sample["query"]])
    retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding[0], k=3)
    retrieved_ids = [doc.metadata["id"] for doc in retrieved_docs]
    
    # Precision@3
    precision = 1.0 if sample["relevant_id"] in retrieved_ids else 0.0
    precision_scores.append(precision)
    
    # Cosine similarity for relevant chunk
    if sample["relevant_id"] in retrieved_ids:
        relevant_doc = next(doc for doc in retrieved_docs if doc.metadata["id"] == sample["relevant_id"])
        relevant_embedding = model.encode([relevant_doc.page_content])
        cosine_score = cosine_similarity(query_embedding, relevant_embedding)[0][0]
        cosine_scores.append(cosine_score)

# Results
print(f"Precision@3: {np.mean(precision_scores):.2f}")
print(f"Average Cosine Similarity: {np.mean(cosine_scores):.2f}")