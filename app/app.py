# File: app.py
import streamlit as st
from generate_answer import generate_answer
from retrieve import retrieve_relevant_chunks

def main():
    st.title("AI Research Q&A System")
    st.write("Ask questions about AI research papers and get answers based on arXiv abstracts.")

    # Input query
    query = st.text_input("Enter your question:", "What are the latest advancements in neural networks?")
    
    if st.button("Get Answer"):
        # Get answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(query)
        
        # Display answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display retrieved chunks
        st.subheader("Retrieved Chunks")
        chunks = retrieve_relevant_chunks(query)
        for chunk in chunks:
            with st.expander(f"From: {chunk['title']}"):
                st.write(f"**Chunk**: {chunk['text']}")
                # st.write(f"**Similarity Score**: {chunk['distance']:.4f}")

if __name__ == "__main__":
    main()