import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Function: Initialize Ollama Embeddings
# ----------------------------------------------------------------------
def create_ollama_embeddings(model_name: str = "nomic-embed-text:v1.5", base_url: str = "http://localhost:11434"):
    log.info(f"Initializing Ollama embeddings model: {model_name}...")
    try:
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        return embeddings
    except Exception as e:
        log.error(f"Error initializing Ollama embeddings: {e}")
        raise RuntimeError(f"Ollama embeddings initialization failed: {e}")

# ----------------------------------------------------------------------
# Function: Custom Document Loader and Parser
# ----------------------------------------------------------------------
def load_and_process_documents(file_path: Path) -> List[Document]:
    """
    Loads the grammar corpus, splits by '//', and extracts level metadata.
    """
    if not file_path.exists():
        log.error(f"File not found: {file_path.resolve()}")
        raise FileNotFoundError(f"File not found: {file_path.resolve()}")

    log.info(f"Loading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the separator '//'
    # The file format is: grammar point block // grammar point block // ...
    raw_chunks = content.split('//')
    
    documents = []
    
    # Regex to find level, e.g., "基礎 第1級", "基礎 第1*級", "進階 第4級"
    # We want to extract the number.
    level_pattern = re.compile(r'(基礎|進階)\s+第(\d+)\*?級')

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Extract level
        match = level_pattern.search(chunk)
        level = None
        if match:
            try:
                level = int(match.group(2))
            except ValueError:
                log.warning(f"Could not parse level number from: {match.group(0)}")
        
        # If no level found, we might skip or assign a default. 
        # For now, let's log a warning and skip if critical, or keep it with level=0.
        if level is None:
            # Try to infer from context or just log. 
            # The corpus seems consistent, so this should be rare if regex is good.
            # log.warning(f"No level found in chunk: {chunk[:50]}...")
            pass
            
        # Clean up the text (remove extra newlines)
        clean_text = re.sub(r'\n+', ' ', chunk).strip()
        
        if clean_text:
            metadata = {"level": level} if level is not None else {}
            documents.append(Document(page_content=clean_text, metadata=metadata))

    log.info(f"Processed {len(documents)} documents.")
    return documents

# ----------------------------------------------------------------------
# Function: Build RAG Vector Store
# ----------------------------------------------------------------------
def build_vector_store(file_path: Path, embedding_model: str = "nomic-embed-text:v1.5", cache_dir: str = "faiss_index"):
    # Use relative path to avoid encoding issues with Chinese characters in absolute path on Windows/FAISS
    # We assume the script is run from the project root or we use the relative path from CWD
    # Go up 1 level: services -> CNtube
    if cache_dir == "faiss_index":
        # Use relative path string instead of absolute path to avoid encoding issues
        cache_dir = os.path.join("..", "faiss_index")
    index_path_str = cache_dir
    
    embeddings = create_ollama_embeddings(embedding_model)

    # Check if index exists
    if os.path.exists(index_path_str):
        log.info(f"Loading existing FAISS vector store from {index_path_str}...")
        try:
            vectorstore = FAISS.load_local(index_path_str, embeddings, allow_dangerous_deserialization=True)
            log.info("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            log.warning(f"Failed to load existing index: {e}. Rebuilding...")

    # If not exists or failed to load, build it
    documents = load_and_process_documents(file_path)
    
    if not documents:
        raise ValueError("No documents loaded to build vector store.")

    log.info("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    log.info("Vector store built successfully.")
    
    # Save to disk
    log.info(f"Saving vector store to {index_path_str}...")
    try:
        vectorstore.save_local(index_path_str)
    except Exception as e:
         log.error(f"Failed to save vector store to {index_path_str}: {e}")
    
    return vectorstore

# ----------------------------------------------------------------------
# Function: Create LLM
# ----------------------------------------------------------------------
def create_llm(model_name: str = "qwen3:1.7b", base_url: str = "http://localhost:11434"):
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        base_url=base_url
    )

# ----------------------------------------------------------------------
# Main Analysis Function
# ----------------------------------------------------------------------
def analyze_grammar_point(transcription: str, user_level: int, grammar_file_path: Path = None):
    """
    Analyzes the transcription using grammar points from the specified user_level.
    """
    # Define paths
    if grammar_file_path is None:
        # Go up 1 level: services -> CNtube
        base_dir = Path(__file__).resolve().parent.parent
        grammar_file = base_dir / "grammar_corpus_cleaned.txt"
    else:
        grammar_file = grammar_file_path
    
    # Build or Load Vector Store (In a real app, we might cache this)
    # For this script, we build it every time or we could save/load. 
    # Building it is fast enough for this size (~7000 lines).
    vectorstore = build_vector_store(grammar_file)
    
    # Create Retriever with filter
    # FAISS retriever supports 'filter' in search_kwargs if the underlying store supports it.
    # LangChain's FAISS implementation handles metadata filtering.
    
    # We need to construct a retriever that applies the filter dynamically.
    # However, standard .as_retriever() with search_kwargs is static.
    # We will use the vectorstore.similarity_search directly in a custom runnable or 
    # use the 'filter' parameter in invoke if supported.
    
    # Actually, the best way in LangChain LCEL is to pass the filter in the config or use a lambda.
    
    llm = create_llm()
    
    prompt_template = """
    作為一位專業的中文語言學家，您的任務是根據提供的文法規則分析以下文章。
    請確保您的分析只基於第 {level} 級出現的文法規則。

    --- 檢索到的文法規則 (Level {level}) ---
    {context}
    -----------------------------------------------

    請根據上述規則，詳細分析以下文章：
    {input}
    
    如果文章中沒有使用到上述規則，請說明「未發現符合程度的文法點」。
    如果有，請列出文法點並解釋。
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Custom retrieval function to handle filtering
    def retrieve_with_filter(inputs):
        query = inputs['input']
        level = inputs['level']
        
        # Perform similarity search with filter
        # Note: FAISS filter format depends on the underlying library, but LangChain wraps it.
        # For LangChain FAISS, we can pass a callable or a dict.
        # Simple dict filter: {'level': level}
        
        log.info(f"Retrieving documents for query: '{query}' at level: {level}")
        
        # Fetch more docs to ensure coverage, then we can let LLM pick or just feed top k
        docs = vectorstore.similarity_search(
            query, 
            k=5, 
            filter={'level': level}
        )
        
        # Format docs for prompt
        return "\n\n".join([d.page_content for d in docs])

    # Build Chain
    chain = (
        RunnableParallel(
            context=retrieve_with_filter,
            input=lambda x: x['input'],
            level=lambda x: x['level']
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    log.info("Executing analysis chain...")
    result = chain.invoke({"input": transcription, "level": user_level})
    return result
