from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from video_processor import extract_audio_from_url
from transcriber import transcribe_audio
from language_analyzer import analyze_text_batch
import traceback
import json

app = Flask(__name__, template_folder="../templates")
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/process', methods=['POST'])
def process_task():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        task = data.get("task")

        if task == "transcribe":
            url = data.get("url")
            if not url:
                return jsonify({"error": "Missing 'url' parameter"}), 400
            
            # 1. Extract Audio & Duration
            audio_path, duration = extract_audio_from_url(url)
            
            # 2. Transcribe (Streaming with Progress)
            def generate_transcription():
                generator = transcribe_audio(audio_path, duration)
                for partial in generator:
                    yield json.dumps(partial) + "\n"
            
            return Response(stream_with_context(generate_transcription()), mimetype='application/x-ndjson')

        elif task == "analyze":
            text = data.get("text")
            if not text:
                return jsonify({"error": "Missing 'text' parameter"}), 400
            
            # 3. Analyze (Streaming with Progress)
            user_level = int(data.get("user_level", 1))
            def generate_analysis():
                generator = analyze_text_batch(text, user_level)
                for partial in generator:
                    yield json.dumps(partial) + "\n"

            return Response(stream_with_context(generate_analysis()), mimetype='application/x-ndjson')

        else:
            return jsonify({"error": f"Unknown task: {task}"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
