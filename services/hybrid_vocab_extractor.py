import json
import os
import re
import random
import jieba
import ollama
from pypinyin import pinyin, Style
from opencc import OpenCC 

# 設定模型 (直接搬過來)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

class HybridVocabExtractor:
    def __init__(self):
        self.cc = OpenCC('s2twp')  # <--- 簡轉
        # --- 1. 原本同學寫的資料庫載入邏輯 ---
        base_dir = os.path.dirname(__file__)
        json_path = os.path.join(base_dir, "coct_words.json")
        
        self.word_db = {}
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.word_db = json.load(f)
                print(f"HybridVocabExtractor: Loaded {len(self.word_db)} words from COCT database.")
        except Exception as e:
            print(f"Error loading word database: {e}")

    # --- 2. 原本同學寫的單字提取邏輯 (Jieba) ---
    def extract_vocab(self, text):
        """接收一段文字，進行斷詞，並回傳分級單字列表"""
        if not text:
            return []

        words = jieba.cut(text)
        extracted_words = {}
        
        for word in words:
            word = word.strip()
            if len(word) < 2 or not re.match(r'[\u4e00-\u9fa5]+', word):
                continue
                
            # 只有當單字在資料庫裡時才列出來
            if word in self.word_db:
                info = self.word_db[word]
                level = info.get("level", 0) if isinstance(info, dict) else info
                
                if word not in extracted_words:
                    extracted_words[word] = {
                        "word": word,
                        "level": level,
                        "frequency": 1
                    }
                else:
                    extracted_words[word]["frequency"] += 1

        # 轉成列表並排序 (按等級高低)
        result_list = list(extracted_words.values())
        result_list.sort(key=lambda x: (x['level'], x['frequency']), reverse=True)
        return result_list

    # --- 3. 新增：原本在 vocabulary_service 裡的語料搜尋 (Helper) ---
    def _search_corpus_example(self, word):
        current_dir = os.path.dirname(__file__)
        # 嘗試找語料庫檔案
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

    # --- 4. 新增：原本在 vocabulary_service 裡的單字卡功能 (Lookup) ---
    def get_word_card(self, word):
        real_example = self._search_corpus_example(word)
        
        if real_example:
            example_instruction = f'I found a corpus sentence: "{real_example}". Use this sentence as the "example" ONLY IF it is a complete, meaningful sentence. If it is fragmented, weird, or Simplified Chinese, REWRITE it into natural Traditional Chinese.'
        else:
            example_instruction = "Create a simple, beginner-friendly sentence using this word in Traditional Chinese."

        # 修改：直接使用 self.word_db 查等級
        official_level = None
        if word in self.word_db:
            info = self.word_db[word]
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
            # 拼音處理
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
            
            # 防呆預設值
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
        
    # --- 5. 新增：原本在 vocabulary_service 裡的句子簡化功能 (Simplify) ---

    def simplify_text(self, text, target_level=2):
        """
        1. 先利用 jieba + word_db 找出句子中「等級 > target_level」的難詞
        2. 將這些難詞明確告訴 LLM，要求它針對性地替換
        """
        
        # --- A. 規則式偵測 (Rule-based Detection) ---
        words = jieba.cut(text)
        detected_hard_words = []
        
        for w in words:
            w = w.strip()
            if w in self.word_db:
                info = self.word_db[w]
                # 取得該字的等級
                w_level = info.get("level", 0) if isinstance(info, dict) else info
                
                # 如果該字等級 > 目標等級，標記為難詞
                if isinstance(w_level, int) and w_level > target_level:
                    detected_hard_words.append(f"{w} (Lv{w_level})")
        
        # 去除重複並轉成字串
        detected_hard_words = list(set(detected_hard_words))
        hard_words_str = ", ".join(detected_hard_words) if detected_hard_words else "None detected (The sentence might already be simple)."

    # --- B. LLM 簡化 (LLM Simplification) ---
    
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
            
            # 確保欄位存在
            if "english_translation" in data and "english_meaning" not in data:
                data["english_meaning"] = data["english_translation"]
            if "detected_hard_words" not in data:
                data["detected_hard_words"] = detected_hard_words
            # --- 新增：OpenCC 強制轉繁體 (加上這段) ---
            # 確保 LLM 吐出的每一個中文字欄位都是繁體，不再有漏網之魚
            if "simplified" in data:
                data["simplified"] = self.cc.convert(data["simplified"])
            
            if "original" in data:
                data["original"] = self.cc.convert(data["original"])

            # 甚至連 changes 裡的建議詞也轉一下比較保險
            if "changes" in data and isinstance(data["changes"], list):
                for item in data["changes"]:
                    if "simple_word" in item:
                        item["simple_word"] = self.cc.convert(item["simple_word"])
            # ----------------------------------------   
            return data
        except Exception as e:
            return {"error": str(e)}