from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8") 
# 運算設定為8位元整數(compute_type="int8") 以節省記憶體 可以視情況改回 float32 
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def transcribe_audio(audio_path: str, duration: float = 0):
    # Returns a generator that yields the full text so far
    segments, _ = model.transcribe(
        audio_path, 
        language="zh",
        beam_size=5,
        # 【重點】暫時切斷上下文依賴！
        condition_on_previous_text=False, 
        
        # --- 額外保險措施 ---
        # 如果模型覺得這段話很爛，原本會直接丟掉，現在我們叫它再試試看
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        compression_ratio_threshold=2.4, # 防止因為聽不懂而開始鬼打牆(重複一樣的字)
    )

    full_text = ""

    for seg in segments:
        text = seg.text.strip()
        start_time = format_time(seg.start)
        
        # Format: [MM:SS] Text
        formatted_line = f"[{start_time}] {text}\n"
        
        full_text += formatted_line
        
        progress = 0
        if duration > 0:
            progress = min(int((seg.end / duration) * 100), 100)
            
        yield {
            "transcript": full_text, 
            "progress": progress,
            "segment": {
                "start": seg.start,
                "end": seg.end,
                "text": text
            }
        }

