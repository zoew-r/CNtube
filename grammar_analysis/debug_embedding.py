import logging
from langchain_community.embeddings import OllamaEmbeddings

# Configure Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def debug_embedding():
    log.info("Initializing Embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5", base_url="http://localhost:11434")
    
    text = "Hello world"
    log.info(f"Embedding text: '{text}'")
    
    try:
        vector = embeddings.embed_query(text)
        log.info(f"Success! Vector length: {len(vector)}")
    except Exception as e:
        log.error(f"Embedding failed: {e}")

if __name__ == "__main__":
    debug_embedding()
