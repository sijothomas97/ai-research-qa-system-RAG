# File: fetch_data.py
import requests
import pandas as pd
import xml.etree.ElementTree as ET

def fetch_arxiv_papers(query="machine learning", max_results=10):
    """
    Fetch AI research paper abstracts from arXiv.
    Args:
        query (str): Search query for arXiv.
        max_results (int): Number of papers to fetch.
    Returns:
        pd.DataFrame: DataFrame with paper titles and abstracts.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from arXiv")
    
    # Parse XML response
    '''
    the Atom namespace (http://www.w3.org/2005/Atom) is used to correctly identify and 
    extract <entry>, <title>, and <summary> elements from the arXiv API’s XML response. 
    It’s a way to tell the parser, “Look for these specific Atom-standard elements,” 
    ensuring accurate data extraction. If you were parsing a different XML API 
    (e.g., RSS or a custom format), you’d use its namespace instead.
    '''
    root = ET.fromstring(response.content) # fromstring() parses XML from a string directly into an Element, which is the root element of the parsed tree.
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        papers.append({"title": title, "abstract": summary.strip()})
    
    return pd.DataFrame(papers)

# Fetch and save data
df = fetch_arxiv_papers(max_results=50)
df.to_csv("../data/arxiv_papers.csv", index=False)
print(f"Fetched {len(df)} papers and saved to arxiv_papers.csv")