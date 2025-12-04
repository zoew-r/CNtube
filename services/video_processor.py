import yt_dlp
import os
import time

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def extract_audio_from_url(video_url: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename_base = f"audio_{timestamp}"
    
    # We provide the base name. yt-dlp/ffmpeg will handle the extensions.
    # If we include .%(ext)s, sometimes it results in double extensions like .mp3.mp3
    # depending on the post-processor behavior.
    # Using just the base name is safer; yt-dlp usually appends the extension if missing,
    # or if we specify it as a literal, ffmpeg will add .mp3 to it.
    # Let's try explicitly using a template that yt-dlp likes, but we will return the expected filename.
    
    # Actually, the most robust way is to use the return value of ydl.download which is not available,
    # or use prepare_filename. But simpler:
    # 1. Define the expected final path
    final_output_path = os.path.join(DOWNLOAD_DIR, f"{filename_base}.mp3")
    
    # 2. Use a template for the temporary download. 
    # We use %(title)s or similar to ensure uniqueness if we weren't using timestamp, 
    # but here we use timestamp.
    # Let's use a specific temp name pattern to avoid collision.
    output_path_template = os.path.join(DOWNLOAD_DIR, f"{filename_base}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path_template,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        
        # Since we use FFmpegExtractAudio with preferredcodec='mp3', 
        # the final file will always have .mp3 extension.
        # prepare_filename returns the original extension (e.g. .webm).
        base, _ = os.path.splitext(filename)
        final_filename = base + ".mp3"
        
        # Verify file exists
        if not os.path.exists(final_filename):
            # Fallback: check if original file exists (conversion failed?)
            if os.path.exists(filename):
                final_filename = filename
            
        duration = info.get('duration', 0)
        return final_filename, duration
