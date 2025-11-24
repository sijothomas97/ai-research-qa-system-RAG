# File: chunk_data.py
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')

def chunk_text(text, max_chunk_length=200):
    """
    Split text into chunks based on sentences, respecting max_chunk_length.
    Args:
        text (str): Input text to chunk.
        max_chunk_length (int): Maximum length of each chunk in characters.
    Returns:
        list: List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_papers(input_file="../data/arxiv_papers.csv", output_file="../data/chunked_papers.csv"):
    """
    Read papers, chunk abstracts, and save to a new CSV.
    """
    df = pd.read_csv(input_file)
    chunked_data = []
    
    for idx, row in df.iterrows():
        chunks = chunk_text(row['abstract'])
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
process_papers()