"""
Transcriber module - Speech to text using Whisper model
"""
from typing import Optional, List, Dict
import whisper


class Transcriber:
    """Handle audio transcription using Whisper model."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcriber with a Whisper model.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
    
    def _load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            self.model = whisper.load_model(self.model_size)
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio to Traditional Chinese text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text in Traditional Chinese, or None if transcription failed
        """
        try:
            self._load_model()
            
            # Transcribe with Chinese language setting
            # Whisper will detect Mandarin and we'll use Traditional Chinese output
            result = self.model.transcribe(
                audio_path,
                language="zh",  # Chinese
                task="transcribe",
                initial_prompt="以下是繁體中文的逐字稿。",  # Prompt for Traditional Chinese
            )
            
            transcription = result.get('text', '')
            
            # Convert to Traditional Chinese if needed
            transcription = self._ensure_traditional_chinese(transcription)
            
            return transcription
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
    
    def _ensure_traditional_chinese(self, text: str) -> str:
        """
        Ensure the text is in Traditional Chinese.
        
        This is a basic implementation. For production, consider using
        a proper Simplified to Traditional Chinese conversion library
        like OpenCC.
        
        Args:
            text: Input text (may be Simplified or Traditional Chinese)
            
        Returns:
            Text converted to Traditional Chinese
        """
        # Common Simplified to Traditional Chinese character mappings
        # This is a basic subset; for production use OpenCC library
        simplified_to_traditional = {
            '国': '國', '说': '說', '这': '這', '时': '時', '学': '學',
            '会': '會', '为': '為', '对': '對', '发': '發', '经': '經',
            '过': '過', '还': '還', '进': '進', '与': '與', '从': '從',
            '现': '現', '开': '開', '关': '關', '无': '無', '问': '問',
            '么': '麼', '们': '們', '头': '頭', '见': '見', '长': '長',
            '门': '門', '点': '點', '义': '義', '电': '電', '动': '動',
            '机': '機', '来': '來', '实': '實', '听': '聽', '话': '話',
            '语': '語', '读': '讀', '写': '寫', '认': '認', '识': '識',
            '词': '詞', '练': '練', '习': '習', '书': '書', '汉': '漢',
            '简': '簡', '体': '體', '传': '傳', '统': '統', '华': '華',
            '请': '請', '让': '讓', '给': '給', '着': '著', '难': '難',
            '双': '雙', '图': '圖', '网': '網', '视': '視', '频': '頻',
        }
        
        result = text
        for simplified, traditional in simplified_to_traditional.items():
            result = result.replace(simplified, traditional)
        
        return result
    
    def get_segments(self, audio_path: str) -> Optional[List[Dict]]:
        """
        Get transcription with timestamps.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of segments with timestamps, or None if transcription failed
        """
        try:
            self._load_model()
            
            result = self.model.transcribe(
                audio_path,
                language="zh",
                task="transcribe",
                initial_prompt="以下是繁體中文的逐字稿。",
            )
            
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': self._ensure_traditional_chinese(segment['text'])
                })
            
            return segments
            
        except Exception as e:
            print(f"Error getting segments: {e}")
            return None
