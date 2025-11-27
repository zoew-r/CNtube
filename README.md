# CNtube 🎬

Learn Traditional Chinese through Video Transcription

CNtube is a web application that helps users learn Chinese by:
1. Extracting audio from video URLs (YouTube, etc.)
2. Transcribing the audio to Traditional Chinese using OpenAI's Whisper model
3. Analyzing the transcription to extract vocabulary and grammar points using LLM

## Features

- **Video URL Processing**: Paste any video URL (YouTube supported) to extract audio
- **Speech-to-Text**: Uses Whisper model for accurate Chinese transcription
- **Traditional Chinese Output**: Ensures transcription is in Traditional Chinese (繁體中文)
- **Vocabulary Extraction**: Identifies key vocabulary with pinyin, English translations, and examples
- **Grammar Analysis**: Extracts grammar points with explanations and example sentences
- **Beautiful UI**: Clean, responsive interface for an enjoyable learning experience

## Requirements

- Python 3.10+
- FFmpeg (for audio extraction)
- OpenAI API Key (for LLM-based vocabulary/grammar analysis)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zoew-r/CNtube.git
cd CNtube
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Paste a video URL (e.g., YouTube link) and click "開始學習" (Start Learning)

4. Wait for the processing to complete:
   - Audio extraction
   - Speech-to-text transcription
   - Vocabulary and grammar analysis

5. Review the results:
   - **逐字稿 (Transcription)**: Full Traditional Chinese transcription
   - **詞彙 (Vocabulary)**: Key words with pinyin, English, and examples
   - **文法點 (Grammar Points)**: Grammar structures with explanations

## Project Structure

```
CNtube/
├── app.py                 # Main Flask application
├── services/
│   ├── __init__.py
│   ├── routes.py          # API routes
│   ├── video_processor.py # Video/audio extraction
│   ├── transcriber.py     # Whisper transcription
│   └── language_analyzer.py # LLM-based analysis
├── templates/
│   └── index.html         # Frontend UI
├── static/                # Static assets
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md
```

## API Endpoints

- `GET /` - Main web interface
- `POST /process` - Process a video URL
  - Request: `{ "video_url": "https://..." }`
  - Response: `{ "success": true, "transcription": "...", "analysis": {...} }`
- `GET /health` - Health check endpoint

## Configuration

Environment variables (set in `.env`):
- `OPENAI_API_KEY`: Your OpenAI API key for LLM analysis
- `FLASK_SECRET_KEY`: Secret key for Flask sessions
- `FLASK_ENV`: Environment mode (development/production)

## Notes

- The Whisper model will be downloaded on first use (~140MB for base model)
- Processing time depends on video length and system resources
- Without an OpenAI API key, mock vocabulary/grammar analysis will be provided
- The app supports any video URL that yt-dlp can handle (YouTube, Vimeo, etc.)

## License

MIT License
