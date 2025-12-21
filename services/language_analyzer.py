import os
import ollama
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

PROMPT = """
You are a professional Taiwanese Mandarin teacher for English speakers.
Your students are learning **Traditional Chinese (Taiwan usage)**, **Zhuyin (Bopomofo)**, and **Hanyu Pinyin**.

Analyze this sentence:
「{sentence}」

Please provide the output in the following JSON format:
{{
  "meaning": "English translation of the sentence",
  "logic": "Explanation of the sentence logic (cause/effect, time, contrast, etc.)",
  "vocabulary": [
    {{
      "word": "Traditional Chinese Word",
      "pinyin": "Hanyu Pinyin (with tone marks)",
      "zhuyin": "Zhuyin (Bopomofo)",
      "definition": "English definition"
    }}
  ],
  "grammar": "Grammar points explanation",
  "misunderstandings": "Common misunderstandings or cultural notes",
  "easy_explanation": "Simple English explanation for beginners"
}}

**Important Requirements:**
1. Use **Traditional Chinese** (Taiwan standard).
2. Ensure **Pinyin** tones are accurate.
3. Include **Zhuyin** for every vocabulary item.
4. Explain grammar simply.
"""


import json
from pypinyin import pinyin, lazy_pinyin, Style

def analyze_sentence(sentence: str) -> dict:
    prompt = PROMPT.format(sentence=sentence)
    
    content = ""

    # Use Ollama
    if USE_OLLAMA:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp["message"]["content"]
    
    # Use OpenAI
    elif client:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = comp.choices[0].message.content
    
    else:
        return {"error": "No LLM client configured"}

    # Parse JSON and fix phonetics
    try:
        # Clean up potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        data = json.loads(content.strip())
        
        # Fix phonetics using pypinyin
        if "vocabulary" in data:
            for item in data["vocabulary"]:
                word = item.get("word", "")
                if word:
                    # Generate Pinyin with tones
                    py_list = pinyin(word, style=Style.TONE)
                    # flatten list like [['nǐ'], ['hǎo']] -> "nǐ hǎo"
                    item["pinyin"] = " ".join([x[0] for x in py_list])
                    
                    # Generate Zhuyin
                    zy_list = pinyin(word, style=Style.BOPOMOFO)
                    item["zhuyin"] = " ".join([x[0] for x in zy_list])
        
        return data

    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        # Return raw content if parsing fails, wrapped in a structure
        return {
            "raw_content": content,
            "error": "Failed to parse JSON"
        }


def analyze_all(asr_data: dict) -> list:
    segments = asr_data.get("segments", [])
    results = []
    for seg in segments:
        text = seg.get("text", "")
        if text:
            analysis = analyze_sentence(text)
            results.append({
                "time": f"{seg['start']} - {seg['end']}",
                "text": text,
                "analysis": analysis
            })
    return results

# Import from the same package
from .grammar_rag_analysis import analyze_grammar_point # adding a dot

def analyze_text_batch(text: str, user_level: int = 1):
    """
    Splits text into lines and analyzes them one by one using RAG grammar analysis, yielding progress.
    """
    # Split by newlines and filter empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    total_lines = len(lines)
    
    if total_lines == 0:
        yield {"progress": 100, "analysis": None}
        return

    for i, line in enumerate(lines):
        # Use RAG analysis
        try:
            rag_result = analyze_grammar_point(line, user_level)
            # Wrap in dict to match expected format
            analysis = {"grammar_analysis": rag_result}
        except Exception as e:
            print(f"Error in RAG analysis: {e}")
            analysis = {"grammar_analysis": f"Error analyzing line: {e}"}

        progress = int(((i + 1) / total_lines) * 100)
        
        yield {
            "progress": progress,
            "analysis": analysis,
            "original_text": line
        }
def get_easier_synonym(word: str):
    """
    接收單詞，回傳簡化後的近義詞資訊。
    實際邏輯委託給 HybridVocabExtractor 處理。
    """
    return vocab_extractor.get_easier_synonym(word)