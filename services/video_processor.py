"""
Video processing module - Extract audio from video URLs
"""
import os
import shutil
import yt_dlp


class VideoProcessor:
    """Handle video downloading and audio extraction."""
    
    def __init__(self):
        """Initialize the video processor."""
        pass
    
    def extract_audio(self, video_url: str, output_dir: str) -> str | None:
        """
        Extract audio from a video URL.
        
        Args:
            video_url: URL of the video (YouTube, etc.)
            output_dir: Directory to save the extracted audio
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        output_path = os.path.join(output_dir, 'audio.mp3')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Check if the file was created
            if os.path.exists(output_path):
                return output_path
            
            # Sometimes yt-dlp creates the file with a different extension
            for ext in ['mp3', 'm4a', 'wav', 'webm']:
                potential_path = os.path.join(output_dir, f'audio.{ext}')
                if os.path.exists(potential_path):
                    return potential_path
            
            return None
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def cleanup(self, temp_dir: str) -> None:
        """
        Clean up temporary files.
        
        Args:
            temp_dir: Directory to clean up
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")
