"""
CNtube - Chinese Learning App through Video Transcription
Main Flask Application
"""
import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
    
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    
    # Register blueprints
    from services.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
