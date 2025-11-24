import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

nltk.download('punkt')

def semantic_chunking(sentences, max_tokens=100, overlap=0.2):
    """
    Group sentences into semantic chunks with overlap.
    Args:
        sentences (list): List of sentences.
        max_tokens (int): Max tokens per chunk.
        overlap (float): Fraction of tokens to overlap.
    Returns:
        list: List of chunked texts.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    overlap_tokens = int(max_tokens * overlap)
    
    for i, sent in enumerate(sentences):
        token_count = len(sent.split())  # Approximate token count
        if current_token_count + token_count <= max_tokens:
            current_chunk.append(sent)
            current_token_count += token_count
        else:
            chunks.append(" ".join(current_chunk))
            # Add overlap by including last sentences
            overlap_sentences = current_chunk[-int(len(current_chunk) * overlap):] if overlap > 0 else []
            current_chunk = overlap_sentences + [sent]
            current_token_count = sum(len(s.split()) for s in current_chunk)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_papers(input_file="arxiv_papers.csv", output_file="chunked_papers.csv"):
    """
    Read papers, chunk abstracts semantically, and save to CSV.
    """
    df = pd.read_csv(input_file)
    chunked_data = []
    
    for idx, row in df.iterrows():
        sentences = sent_tokenize(row['abstract'])
        chunks = semantic_chunking(sentences, max_tokens=100, overlap=0.2)
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "paper_id": idx,
                "title": row['title'],
                "chunk_id": f"{idx}_{i}",
                "chunk_text": chunk
            })
    
    chunked_df = pd.DataFrame(chunked_data)
    chunked_df.to_csv(output_file, index=False)
    print(f"Chunked {len(chunked_df)} chunks and saved to {output_file}")

# Run chunking
if __name__ == "__main__":
    process_papers()