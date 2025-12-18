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

        # --- 額外添加的東西 --- 
        condition_on_previous_text=False, 
        # 【重點】暫時切斷上下文依賴！
        no_speech_threshold=0.95,
        # 雜訊容錯設定0.95
        # 如果模型覺得這段話很爛，原本會直接丟掉，現在我們叫它再試試看
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        compression_ratio_threshold=2.4, # 防止因為聽不懂而開始鬼打牆(重複一樣的字)
    )

    full_text = ""

    for seg in segments:
        text = seg.text.strip()
        start_time = format_time(seg.start)

        # --- 新增：生成注音配對 ---
        # 1. 取得注音列表 (BOPOMOFO 代表注音符號)
        # heteronym=False 代表不顯示多音字的所有發音，只取最常用的
        zhuyin_list = pinyin(text, style=Style.BOPOMOFO)
        
        # 2. 將文字與注音配對
        # 結果會變成像這樣：[{"char": "你", "zy": "ㄋㄧˇ"}, {"char": "好", "zy": "ㄏㄠˇ"}]
        # zip 用來把兩個列表鎖在一起
        chars_with_zhuyin = []
        for char, zy_item in zip(text, zhuyin_list):
            # zy_item 是一個 list，例如 ['ㄋㄧˇ']，我們取第一個
            zy = zy_item[0] 
            # 如果是標點符號，pinyin 會原樣回傳，我們就把注音設為空字串，避免顯示在上方
            if char == zy: 
                zy = "" 
            chars_with_zhuyin.append({"char": char, "zy": zy})
        # ------------------------
        
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

