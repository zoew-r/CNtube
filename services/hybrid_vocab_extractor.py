import json
import os
import jieba
import re

class HybridVocabExtractor:
    def __init__(self):
        # 載入單字分級資料庫
        base_dir = os.path.dirname(__file__)
        json_path = os.path.join(base_dir, "coct_words.json")
        
        self.word_db = {}
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.word_db = json.load(f)
                print(f"Loaded {len(self.word_db)} words from COCT database.")
        except Exception as e:
            print(f"Error loading word database: {e}")

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
                
            # 只有當單字在資料庫裡時才列出來 (或是你可以選擇全部列出)
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