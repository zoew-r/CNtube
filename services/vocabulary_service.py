import os
import ollama
import json
import random
import re
from pypinyin import pinyin, Style

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

COCT_DB = {}
try:
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "coct_words.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            COCT_DB = json.load(f)
        print(f"Vocab Service: Loaded {len(COCT_DB)} words from COCT database.")
    else:
        print("Vocab Service: coct_words.json not found.")
except Exception as e:
    print(f"Vocab Service: Error loading COCT DB: {e}")

# --- Helper: Search Corpus (Unchanged) ---
def search_corpus_example(word):
    current_dir = os.path.dirname(__file__)
    corpus_files = [
        os.path.join(current_dir, "grammar_corpus_cleaned.txt"),
        os.path.join(current_dir, "..", "grammar_analysis", "grammar_corpus_cleaned.txt")
    ]
    corpus_path = None
    for p in corpus_files:
        if os.path.exists(p):
            corpus_path = p
            break
    if not corpus_path: return None

    found_sentences = []
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = content.split('//')
        for block in blocks:
            if word not in block: continue
            lines = block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if len(line) < 6 or len(line) > 60: continue
                if "第" in line and "級" in line: continue 
                if line.isdigit(): continue 
                if not re.search(r'[。！？]$', line): continue
                if re.match(r'^\d+\.', line) or re.match(r'^[A-Z]\.', line): continue
                if word in line:
                    clean_line = re.sub(r'^[A-Z]：', '', line)
                    if word in clean_line: found_sentences.append(clean_line)
        if found_sentences:
            ideal = [s for s in found_sentences if 8 <= len(s) <= 30]
            return random.choice(ideal) if ideal else random.choice(found_sentences)
    except: pass
    return None

# --- Get Word Card (Unchanged) ---
def get_word_card(word):
    real_example = search_corpus_example(word)
    
    if real_example:
        example_instruction = f'I found a corpus sentence: "{real_example}". Use this sentence as the "example" ONLY IF it is a complete, meaningful sentence. If it is fragmented, weird, or Simplified Chinese, REWRITE it into natural Traditional Chinese.'
    else:
        example_instruction = "Create a simple, beginner-friendly sentence using this word in Traditional Chinese."

    official_level = None
    if word in COCT_DB:
        info = COCT_DB[word]
        lvl = info.get("level") if isinstance(info, dict) else info
        official_level = f"Level {lvl}"
    
    if official_level:
        level_instruction = f'- "level": Set to "{official_level}" (Strict official level).'
    else:
        level_instruction = '- "level": Estimate TOCFL level.'

    prompt = f"""
    You are a professional Chinese language teacher.
    Create a vocabulary card for: 「{word}」.
    
    {example_instruction}
    
    **STRICT CONSTRAINT**: 
    1. All Chinese output MUST be in **Traditional Chinese (繁體中文)**.
    2. Ensure "example_en" is provided.
    
    You must return strict JSON with these keys:
    - "word": The Chinese word.
    {level_instruction}
    - "translation": The English translation.
    - "definition_en": English definition.
    - "definition_ch": Traditional Chinese definition.
    - "example": The example sentence (Traditional Chinese).
    - "example_en": English translation of the example.
    - "simpler_synonym": A simpler synonym (Traditional Chinese) or null.
    - "simpler_synonym_pinyin": Pinyin for the synonym.
    """

    try:
        py = pinyin(word, style=Style.TONE)
        pinyin_str = " ".join([x[0] for x in py])
        zy = pinyin(word, style=Style.BOPOMOFO)
        zhuyin_str = " ".join([x[0] for x in zy])

        resp = ollama.chat(
            model=OLLAMA_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            format="json",
            options={"temperature": 0.2}
        )
        content = resp['message']['content']
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
            
        data = json.loads(content)
        
        defaults = {
            "level": official_level if official_level else "General",
            "translation": word,
            "definition_en": "Definition not available.",
            "definition_ch": "暫無釋義",
            "example": "暫無例句",
            "example_en": "No example translation available.",
            "simpler_synonym": None,
            "simpler_synonym_pinyin": ""
        }
        for key, default_val in defaults.items():
            if key not in data or not data[key]:
                data[key] = default_val

        data["word"] = word
        data["pinyin"] = pinyin_str
        data["zhuyin"] = zhuyin_str
        
        return data

    except Exception as e:
        return {"word": word, "error": str(e)}

# --- Simplify Text (修正重點在此) ---
def simplify_text(text):
    prompt = f"""
    You are a **Taiwanese Mandarin Teacher (台灣華語教師)** who is also fluent in English.
    
    Task: Rewrite the following sentence for a Level 2 student (beginner).
    Target Sentence: "{text}"
    
    **TEACHER'S GUIDELINES**: 
    1. **Authenticity**: Use **Traditional Chinese (繁體中文)** and **Taiwanese vocabulary** (e.g., use "計程車" not "出租車", "影片" not "視頻").
    2. **Meaning**: Keep the original meaning. Do NOT change entities (e.g., keep "檢方/Prosecutors", do not change to "警方/Police").
    3. **Translation**: You MUST provide an English translation for the simplified sentence.
    
    Return strict JSON:
    {{
        "original": "{text}",
        "simplified": "The simplified sentence in Traditional Chinese",
        "english_translation": "The English translation of the SIMPLIFIED sentence",
        "changes": [ {{"hard_word": "original", "simple_word": "simple"}} ]
    }}
    """
    
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            format="json", 
            options={"temperature": 0.2}
        )
        content = resp['message']['content']
        if "```json" in content: content = content.split("```json")[1].split("```")[0]
        
        data = json.loads(content)
        
        # Mapping 舊前端欄位以防萬一
        if "english_translation" in data and "english_meaning" not in data:
            data["english_meaning"] = data["english_translation"]
            
        return data
    except Exception as e:
        return {"error": str(e)}