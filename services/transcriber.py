from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu")

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def transcribe_audio(audio_path: str, duration: float = 0):
    # Returns a generator that yields the full text so far
    segments, _ = model.transcribe(audio_path, beam_size=5)

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

