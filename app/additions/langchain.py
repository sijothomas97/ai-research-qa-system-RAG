from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer

# Initialize components
model = SentenceTransformer("all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(
    ["Order 123: Shipped on 07/10/2025", "Order 124: Delivered on 07/12/2025"],
    model.encode,
    metadatas=[{"id": 0}, {"id": 1}]
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = OpenAI(model_name="gpt-3.5-turbo")

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query
query = "Where’s my order 123?"
response = qa_chain.run(query)
print(response)  # Output: "Order 123 was shipped on July 10, 2025."