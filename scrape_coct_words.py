import requests
import json
import os

# URL for the TOCFL word list JSON
JSON_URL = "https://raw.githubusercontent.com/PSeitz/tocfl/main/tocfl_words.json"
OUTPUT_FILE = "coct_words.json"

def download_and_process_json():
    print(f"Downloading JSON from {JSON_URL}...")
    try:
        response = requests.get(JSON_URL)
        response.raise_for_status()
        
        # The file is NDJSON (newline delimited JSON)
        content = response.text
        lines = content.strip().split('\n')
        
        vocab_db = {}
        count = 0
        
        print("Processing vocabulary data...")
        for line in lines:
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                
                word = item.get('text')
                if not word:
                    continue
                    
                pinyin = item.get('pinyin', "")
                zhuyin = item.get('zhuyin', "")
                level = item.get('tocfl_level', 1)
                
                # Note: This source lacks English definitions.
                # We will rely on the LLM to fill this in during extraction,
                definition = "" 
                
                if word not in vocab_db:
                    vocab_db[word] = {
                        "pinyin": pinyin,
                        "zhuyin": zhuyin,
                        "definition": definition,
                        "level": level
                    }
                    count += 1
            except json.JSONDecodeError:
                continue
        
        print(f"Processed {count} words.")
        
        # Save to JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(vocab_db, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_and_process_json()
