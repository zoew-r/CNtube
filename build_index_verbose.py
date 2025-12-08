import os
import re
import logging
import time
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_and_process_documents(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    raw_chunks = content.split('//')
    documents = []
    level_pattern = re.compile(r'(基礎|進階)\s+第(\d+)\*?級')

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk: continue
        match = level_pattern.search(chunk)
        level = None
        if match:
            try:
                level = int(match.group(2))
            except: pass
        
        clean_text = re.sub(r'\n+', ' ', chunk).strip()
        if clean_text:
            metadata = {"level": level} if level is not None else {}
            documents.append(Document(page_content=clean_text, metadata=metadata))
    return documents

def build_verbose():
    base_dir = Path(".").resolve()
    grammar_file = base_dir / "grammar_analysis" / "grammar_corpus_cleaned.txt"
    index_path = "faiss_index"
    
    log.info("Loading Documents...")
    docs = load_and_process_documents(grammar_file)
    log.info(f"Loaded {len(docs)} documents.")
    
    log.info("Initializing Embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5", base_url="http://localhost:11434")
    
    # We will embed in batches to show progress
    # FAISS.from_documents can take all, but we want to see progress.
    # Actually, we can just use FAISS.from_documents and hope, 
    # BUT let's do manual embedding to be sure.
    
    # Unfortuantely FAISS.from_documents handles the embedding call internally.
    # We can embed first, then create store.
    
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    
    log.info("Starting embedding process (custom batching)...")
    batch_size = 10
    total = len(texts)
    
    # Create empty FAISS
    # To do this, we need at least one embedding.
    # Let's do the first batch
    
    try:
        # embed_documents is the method
        # But constructing FAISS from pre-embedded vectors is different. 
        # Easier: just use from_documents on smaller batches and merge?
        # Or just use from_documents on the whole thing but knowing it calls embed_documents.
        
        # Let's try batching:
        vectorstore = None
        
        for i in range(0, total, batch_size):
            batch_docs = docs[i : i + batch_size]
            log.info(f"Processing batch {i} to {min(i+batch_size, total)} of {total}...")
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch_docs, embeddings)
            else:
                vectorstore.add_documents(batch_docs)
            
            # Optional: save intermediate? No need.
            
        log.info("Embedding complete.")
        log.info(f"Saving to {index_path}...")
        vectorstore.save_local(index_path)
        log.info("Done.")
        
    except Exception as e:
        log.error(f"Failed: {e}")

if __name__ == "__main__":
    build_verbose()
