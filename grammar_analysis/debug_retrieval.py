import os
import logging
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Configure Logging
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

def debug_index_stats():
    index_path = "faiss_index"
    print(f"Loading Index from {index_path}...", flush=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5", base_url="http://localhost:11434")
    
    try:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # Hack to access internal docstore if possible, or just search generic
    # FAISS in langchain doesn't easily expose all docs. 
    # But we can search for "." or something common with a large k.
    
    print("Searching for specific Level 1 text '我是'...", flush=True)
    docs = vectorstore.similarity_search("我是", k=50, filter={'level': 1})
    print(f"Found {len(docs)} documents for '我是' with level=1 filter.")
    for i, d in enumerate(docs[:3]):
        print(f"Match {i}: {d.page_content[:50]}...")

    print("\nCheck generic distribution (searching '。' with k=100)...", flush=True)
    docs_all = vectorstore.similarity_search("。", k=100)
    level_counts = {}
    for d in docs_all:
        l = d.metadata.get('level', 'None')
        level_counts[l] = level_counts.get(l, 0) + 1
    
    print("Level distribution in top 100 results:", level_counts)

if __name__ == "__main__":
    debug_index_stats()
