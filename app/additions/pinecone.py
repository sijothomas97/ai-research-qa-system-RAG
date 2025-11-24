import pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Create or connect to an index
index_name = "products"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")  # 384 for all-MiniLM-L6-v2
index = pinecone.Index(index_name)

# Initialize Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Upsert product embeddings
products = ["Phone X: 12-hour battery life", "Phone Y: 8-hour battery life"]
embeddings = model.encode(products)
index.upsert(vectors=[
    (str(i), emb, {"id": i, "category": "phone"}) for i, emb in enumerate(embeddings)
])

# Query for relevant products
query = "Phone with long battery life"
query_embedding = model.encode([query])[0]
results = index.query(query_embedding, top_k=2, include_metadata=True)

# Output results
for match in results["matches"]:
    print(f"Product ID: {match['metadata']['id']}, Score: {match['score']}")
# Example Output: Product ID: 0, Score: 0.92 (Phone X)