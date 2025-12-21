import os
import re
import json
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
from pypinyin import pinyin, Style

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GRAMMAR_MODEL = os.getenv("GRAMMAR_MODEL", "qwen2.5:7b") 

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
    if cache_dir == "faiss_index":
        # Check if index exists in current dir (e.g. running from CNtube)
        if os.path.exists("faiss_index"):
            cache_dir = "faiss_index"
        # Check if index exists in parent dir (e.g. running from services)
        elif os.path.exists(os.path.join("..", "faiss_index")):
            cache_dir = os.path.join("..", "faiss_index")
        else:
            # Default for creation: if running from services, put it in parent
            if os.path.basename(os.getcwd()) == "services":
                cache_dir = os.path.join("..", "faiss_index")
            else:
                cache_dir = "faiss_index"
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
def create_llm(model_name: str = GRAMMAR_MODEL, base_url: str = "http://localhost:11434"):
    log.info(f"Creating LLM with model: {model_name}")
    # Higher temperature for JSON creativity? No, low for formatting.
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        format="json",  # Force JSON mode in Ollama
        base_url=base_url
    )

# ----------------------------------------------------------------------
# Main Analysis Function
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Global Cache for Chain
# ----------------------------------------------------------------------
_CACHED_CHAIN = None

def get_rag_chain(grammar_file_path: Path):
    """
    Creates or returns the cached RAG chain.
    """
    global _CACHED_CHAIN
    
    if _CACHED_CHAIN is not None:
        return _CACHED_CHAIN
        
    log.info("Initializing RAG Chain (Loading Models)...")
    
    # 1. Vector Store
    vectorstore = build_vector_store(grammar_file_path)
    
    # 2. LLM
    llm = create_llm()
    
    # 3. Prompt (JSON focused)
    prompt_template = """
    You are a **English-Speaking Professor of Chinese Linguistics**. 
    Your native language is English, and you explain Chinese grammar concepts to English speakers.
    
    Your task is to analyze the "Target Sentence" using **STRICTLY AND ONLY** the provided "Retrieved Grammar Rules".

    --- Retrieved Grammar Rules (Level {level}) ---
    {context}
    -----------------------------------------------

    Target Sentence: "{input}"

    **PROFESSOR'S RULES**:
    1. **Strict Matching**: If the Target Sentence does NOT clearly demonstrate a rule listed above, admit it. Set "found": false. Do not hallucinate connections.
    2. **Language Protocol**: 
       - **english_translation_of_sentence**: Must be in natural **English**.
       - **explanation_in_english**: Explain the grammar logic in **English** (as if lecturing to students).
       - **grammar_point_cn**: Use **Traditional Chinese** characters for the rule name.

    Return JSON strictly:
    {{
      "english_translation_of_sentence": "English translation here...",
      "matched_grammar": {{
          "found": true or false, 
          "level": {level},
          "grammar_point_cn": "Pattern Name (Traditional Chinese)",
          "explanation_in_english": "Detailed explanation in English..."
      }},
      "additional_info": {{
          "point": "Any other key point (Traditional Chinese)",
          "explanation": "Brief note (English)"
      }}
    }}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 4. Retriever
    def retrieve_with_filter(inputs):
        query = inputs['input']
        level = inputs['level']
        
        log.info(f"Retrieving documents for query: '{query}' at level: {level}")
        docs = vectorstore.similarity_search(query, k=5, filter={'level': level})
        return "\n\n".join([d.page_content for d in docs])

    # 5. Chain
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
    
    _CACHED_CHAIN = chain
    log.info("RAG Chain initialized and cached.")
    return chain

# ----------------------------------------------------------------------
# Main Analysis Function
# ----------------------------------------------------------------------
def analyze_grammar_point(transcription: str, user_level: int, grammar_file_path: Path = None):
    """
    Analyzes the transcription using RAG, requesting JSON, then formats it nicely.
    """
    if grammar_file_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        grammar_file = base_dir / "grammar_analysis" / "grammar_corpus_cleaned.txt"
    else:
        grammar_file = grammar_file_path
    
    chain = get_rag_chain(grammar_file)
    
    # 1. Generate Phonetics locally
    try:
        py_list = pinyin(transcription, style=Style.TONE)
        pinyin_str = " ".join([x[0] for x in py_list])
        
        zy_list = pinyin(transcription, style=Style.BOPOMOFO)
        zhuyin_str = " ".join([x[0] for x in zy_list])
    except Exception as e:
        log.error(f"Error generating phonetics: {e}")
        pinyin_str = "Error"
        zhuyin_str = "Error"

    # 2. Get JSON from LLM
    log.info("Executing analysis chain...")
    raw_json_str = chain.invoke({"input": transcription, "level": user_level})
    
    # 3. Parse JSON
    try:
        # Clean up any potential markdown backticks
        cleaned_json = raw_json_str.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[7:]
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3]
        
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse LLM JSON: {e}. Content: {raw_json_str}")
        return f"Error: Could not parse analysis results. Raw output:\n{raw_json_str}"

    # 4. Construct Final Formatted String
    translation = data.get("translation", "")
    
    matched = data.get("matched_grammar", {})
    found = matched.get("found", False)
    level = matched.get("level", user_level)
    point = matched.get("point", "Unknown")
    explanation = matched.get("explanation", "")
    
    additional = data.get("additional_info", {})
    add_point = additional.get("point", "")
    add_exp = additional.get("explanation", "")
    
    # Build Output
    output = []
    output.append("1. **Sentence**:")
    output.append(f"   - English Translation: {translation}")
    output.append(f"   - **Zhuyin (Bopomofo)**: {zhuyin_str}")
    output.append(f"   - Hanyu Pinyin: {pinyin_str}")
    
    output.append("2. **Grammar Explanation**:")
    if found:
        output.append(f"   - Level: {level}")
        output.append(f"   - Point: {point}")
        output.append(f"   - Explanation: {explanation}")
    else:
        output.append(f"   - No matching grammar points found for Level {user_level}.")
        
    if add_point and add_point.lower() != "none":
        output.append("3. **Additional Information**:")
        output.append(f"   - Point: {add_point}")
        if add_exp:
            output.append(f"   - Explanation: {add_exp}")

    return "\n".join(output)
