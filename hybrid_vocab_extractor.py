import json
import jieba
import os
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

class HybridVocabExtractor:
    def __init__(self, vocab_file="coct_words.json"):
        self.vocab_db = self._load_vocab(vocab_file)
        self.llm = self._init_llm()
        
    def _load_vocab(self, vocab_file):
        if not os.path.exists(vocab_file):
            print(f"Warning: {vocab_file} not found. Local lookup will fail.")
            return {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _init_llm(self):
        # Use Ollama by default, or OpenAI if key is present
        if os.getenv("USE_OLLAMA", "true").lower() == "true":
            return ChatOllama(model="gemma2:2b", format="json")
        else:
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    def extract_vocab(self, text, mode="hybrid"):
        """
        Extracts vocabulary from text.
        Modes:
        - local: Only use local DB (fast, no examples).
        - hybrid: Local DB for selection + LLM for examples/definitions.
        - llm: LLM does everything (slower).
        """
        
        # 1. Tokenize and Filter (Local Step)
        words = jieba.lcut(text)
        unique_words = set(words)
        
        extracted_candidates = []
        
        for word in unique_words:
            if word in self.vocab_db:
                info = self.vocab_db[word]
                # Filter out very basic words (optional, maybe keep all for now or filter level 1)
                # Let's keep all found words but categorize them.
                extracted_candidates.append({
                    "word": word,
                    "pinyin": info["pinyin"],
                    "zhuyin": info.get("zhuyin", ""),
                    "level": info["level"],
                    "definition": info["definition"] # Might be empty
                })
        
        # Sort by Level (Foundational 1-2, Intermediate 3-4, Advanced 5+)
        
        # If mode is 'local', return immediately
        if mode == "local":
            return self._group_by_level(extracted_candidates)
            
        # 2. Hybrid Step: Enrich with LLM
        # User requested to limit to 10 words per level.
        
        # Group by level first
        grouped = {"foundational": [], "intermediate": [], "advanced": []}
        for item in extracted_candidates:
            lvl = item['level']
            if lvl <= 2:
                grouped["foundational"].append(item)
            elif lvl <= 4:
                grouped["intermediate"].append(item)
            else:
                grouped["advanced"].append(item)
        
        # Select top 10 from each level (assuming input is already somewhat sorted or random, 
        # but we can sort by level desc within group if needed, though level is same/similar)
        target_words = []
        for key in grouped:
            # Sort by level descending (though within group variance is small) just in case
            sorted_group = sorted(grouped[key], key=lambda x: x['level'], reverse=True)
            target_words.extend(sorted_group[:10])
        
        if not target_words:
            return {"foundational": [], "intermediate": [], "advanced": []}

        # Batch processing to avoid LLM limits
        BATCH_SIZE = 15
        enriched_data = []
        
        for i in range(0, len(target_words), BATCH_SIZE):
            batch = target_words[i:i+BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1} of {(len(target_words)-1)//BATCH_SIZE + 1}...")
            batch_result = self._enrich_with_llm(batch)
            enriched_data.extend(batch_result)
        
        # Merge enriched data back
        final_results = []
        for item in target_words: # Use sorted order
            # Check if we have enriched version
            enriched = next((x for x in enriched_data if x['word'] == item['word']), None)
            if enriched:
                # Ensure level is preserved if LLM didn't return it (though _enrich_with_llm handles it)
                if 'level' not in enriched:
                    enriched['level'] = item['level']
                final_results.append(enriched)
            else:
                # Keep original (local) data if enrichment failed
                final_results.append(item)
                
        return self._group_by_level(final_results)

    def _enrich_with_llm(self, words_data):
        """
        Uses LLM to generate definitions and examples for a list of words.
        """
        word_list = [w['word'] for w in words_data]
        
        # Simplified prompt to ensure JSON validity and focus
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Chinese language teacher. 
            For the following Chinese words: {words}
            
            Provide a JSON output with the following structure for EACH word:
            [
                {{
                    "word": "the word",
                    "pinyin": "pinyin with tone marks",
                    "zhuyin": "zhuyin symbols",
                    "definition": "English definition",
                    "example": "A simple Chinese example sentence using the word.",
                    "translation": "English translation of the example sentence."
                }},
                ...
            ]
            
            Ensure the JSON is valid. Return ONLY the JSON list.
            """
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            # print(f"Calling LLM to enrich {len(word_list)} words...")
            result = chain.invoke({"words": ", ".join(word_list)})
            
            # Merge level info back from input
            for res_item in result:
                original = next((w for w in words_data if w['word'] == res_item['word']), None)
                if original:
                    res_item['level'] = original['level']
            
            return result
        except Exception as e:
            print(f"LLM Error: {e}")
            return words_data # Fallback to input data

    def _group_by_level(self, items):
        grouped = {
            "foundational": [], # Level 1-2
            "intermediate": [], # Level 3-4
            "advanced": []      # Level 5+
        }
        
        for item in items:
            lvl = item.get('level', 1)
            if lvl <= 2:
                grouped["foundational"].append(item)
            elif lvl <= 4:
                grouped["intermediate"].append(item)
            else:
                grouped["advanced"].append(item)
                
        return grouped
