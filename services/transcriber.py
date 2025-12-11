from faster_whisper import WhisperModel
from pypinyin import pinyin, Style
from opencc import OpenCC
import os
import json
import ollama
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

model_size = "medium"  # Upgraded from "small" for better multilingual support
# Run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")
# Initialize OpenCC for Simplified to Traditional (Taiwan) conversion (with phrases)
cc = OpenCC('s2twp')

CORRECTION_PROMPT = """
You are an expert editor for Taiwanese Mandarin transcripts with deep contextual understanding.

Your task: Analyze the sentence and fix errors in LOGIC, MEANING, and GRAMMAR using CONTEXT from surrounding sentences.

CRITICAL HANDLING RULES:
1. **Pure English sentences** → Return EXACTLY as is, do NOT modify, do NOT translate
2. **Mixed Chinese/English** → Fix Chinese errors, keep ALL English words unchanged
3. **Pure Chinese sentences** → Fix homophones, logic, grammar using context

Common issues to fix (Chinese only):
- Homophones (同音異字): "遠算法" → "演算法", "音該" → "應該"
- Logic errors (語意錯誤): "他說他不去了，所以他會去" → "他說他不去了，所以他不會去"
- Grammar (文法): "我有很多的問題" → "我有很多問題"
- Context-based corrections: Use前後文to understand the intended meaning

Examples:
"科技巨頭 Google 推出了 AI 遠算法" → "科技巨頭 Google 推出了 AI 演算法"
"This is a demo of the new feature" → "This is a demo of the new feature"
"Apple 發布了新的 iPhone，它的鏡頭非常的清楚" → "Apple 發布了新的 iPhone，它的鏡頭非常清楚"

ABSOLUTE RULES:
1. **NEVER translate English to Chinese**
2. **NEVER summarize or skip content** - Output everything
3. **Use ONLY Traditional Chinese (繁體中文)** for Chinese text
4. **No explanations, just output the corrected text**
5. **Use context to resolve ambiguous homophones**

Previous sentence: {prev}
Current sentence to correct: {text}
Next sentence: {next}

Only output the corrected CURRENT sentence, do not include previous or next sentences in your response.
"""

def correct_transcript(text: str, prev_text: str = "", next_text: str = "") -> str:
    """
    Corrects Chinese homophones in the text using LLM with context, preserving English.
    """
    if not text.strip():
        return text
    
    # Check if text is pure English (or mostly English)
    # Count Chinese characters vs total characters
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len([c for c in text if c.strip() and not c.isspace()])
    
    # If less than 10% Chinese, treat as English and skip LLM
    if total_chars > 0 and (chinese_chars / total_chars) < 0.1:
        print(f"[DEBUG] Detected pure English, skipping LLM: {text[:50]}...")
        return text  # Return as-is for pure English

    prompt = CORRECTION_PROMPT.format(
        text=text,
        prev=prev_text if prev_text else "(沒有前一句)",
        next=next_text if next_text else "(沒有下一句)"
    )
    content = text # Default to original if fails

    try:
        if USE_OLLAMA:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp["message"]["content"].strip()
        elif client:
            comp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            content = comp.choices[0].message.content.strip()
        
        # Clean up any prompt artifacts that LLM might include
        # Remove common prefixes that shouldn't be in the output
        prefixes_to_remove = ["Output:", "output:", "Result:", "Corrected:", "corrected:"]
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
            
        # Basic validation: If result is wildly different length, probably hallucinated
        if abs(len(content) - len(text)) > len(text) * 0.5:
             return text
             
        return content
        
    except Exception as e:
        print(f"Error correcting transcript: {e}")
        return text

def format_time(seconds):
    """Converts seconds to MM:SS format"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def transcribe_audio(audio_path: str, duration: float = 0):
    # Enhanced multilingual detection with word-level analysis
    # word_timestamps: Analyze language per word, not just per segment
    # VAD parameters: Create longer, more natural sentence segments
    segments, info = model.transcribe(
        audio_path, 
        beam_size=5,
        task="transcribe",
        word_timestamps=True,
        # VAD (Voice Activity Detection) tuning for longer sentences
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=2000,  # Much longer pause needed (2 seconds)
            speech_pad_ms=500,  # More padding
            max_speech_duration_s=60  # Allow up to 60s continuous speech
        )
        # No language lock - let Whisper detect per word
    )
    
    # Debug: Print detected language
    print(f"[DEBUG] Detected primary language: {info.language} (confidence: {info.language_probability:.2f})")

    full_text = ""
    
    # First pass: collect all segments and convert to Traditional Chinese
    all_segments = []
    segment_count = 0
    for seg in segments:
        text = seg.text.strip()
        # Force Traditional Chinese Conversion
        text = cc.convert(text)
        all_segments.append({
            "text": text,
            "start": seg.start,
            "end": seg.end
        })
        
        # Progress update during collection (yield dict, not JSON string)
        segment_count += 1
        if segment_count % 5 == 0:  # Update every 5 segments
            yield {
                "transcript": "",  # No transcript yet during collection
                "progress": 5,  # Low progress during collection phase
                "segment": None
            }
    
    
    # Second pass: process with context
    for i, seg_info in enumerate(all_segments):
        text = seg_info["text"]
        prev_text = all_segments[i-1]["text"] if i > 0 else ""
        next_text = all_segments[i+1]["text"] if i < len(all_segments) - 1 else ""
        
        print(f"[DEBUG] Whisper raw output: {text}")
        print(f"[DEBUG] After OpenCC #1: {text}")
        
        # 2. Correct homophones using specific LLM call WITH CONTEXT
        text = correct_transcript(text, prev_text, next_text)
        print(f"[DEBUG] After LLM correction: {text}")
        
        # 3. Ensure LLM output is also Traditional Chinese (double-check)
        text = cc.convert(text)
        print(f"[DEBUG] After OpenCC #2: {text}")
        print(f"[DEBUG] Final output: {text}\n")
        
        # Generate Zhuyin Ruby Text
        # Process character by character to ensure perfect alignment
        ruby_text = []
        if text:
            for char in text:
                # Check if character is Chinese (basic CJK Unified Ideographs range)
                if '\u4e00' <= char <= '\u9fff':
                    # Generate Zhuyin for this single character
                    # pinyin returns a list of lists, take the first one
                    z_list = pinyin(char, style=Style.BOPOMOFO)
                    if z_list and z_list[0]:
                        ruby_text.append({"char": char, "zhuyin": z_list[0][0]})
                    else:
                        ruby_text.append({"char": char, "zhuyin": ""})
                else:
                    # Non-Chinese character (English, punctuation, number, etc.)
                    # No Zhuyin needed
                    ruby_text.append({"char": char, "zhuyin": ""})

        # Format: [MM:SS] Text (matching original format)
        start_time = format_time(seg_info['start'])
        formatted_line = f"[{start_time}] {text}\n"
        full_text += formatted_line
        
        progress = 0
        if duration > 0:
            progress = min(int((seg_info["end"] / duration) * 100), 100)
            
        yield {
            "transcript": full_text, 
            "progress": progress,
            "segment": {
                "start": seg_info["start"],
                "end": seg_info["end"],
                "text": text,
                "ruby_text": ruby_text
            }
        }
