"""
Flask routes for CNtube application
"""
import os
import uuid
from flask import Blueprint, render_template, request, jsonify

from services.video_processor import VideoProcessor
from services.transcriber import Transcriber
from services.language_analyzer import LanguageAnalyzer

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@main_bp.route('/process', methods=['POST'])
def process_video():
    """
    Process a video URL:
    1. Extract audio from video
    2. Transcribe audio to Traditional Chinese
    3. Analyze vocabulary and grammar
    """
    data = request.get_json()
    
    if not data or 'video_url' not in data:
        return jsonify({'error': 'Video URL is required'}), 400
    
    video_url = data['video_url']
    
    try:
        # Generate unique session ID for this processing job
        session_id = str(uuid.uuid4())
        temp_dir = os.path.join('temp', session_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Step 1: Extract audio from video
        video_processor = VideoProcessor()
        audio_path = video_processor.extract_audio(video_url, temp_dir)
        
        if not audio_path:
            return jsonify({'error': 'Failed to extract audio from video'}), 500
        
        # Step 2: Transcribe audio to Traditional Chinese
        transcriber = Transcriber()
        transcription = transcriber.transcribe(audio_path)
        
        if not transcription:
            return jsonify({'error': 'Failed to transcribe audio'}), 500
        
        # Step 3: Analyze vocabulary and grammar
        analyzer = LanguageAnalyzer()
        analysis = analyzer.analyze(transcription)
        
        # Clean up temporary files
        video_processor.cleanup(temp_dir)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})
