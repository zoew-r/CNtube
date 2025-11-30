#!/bin/bash

# CNtube Startup Script

echo "🚀 Starting CNtube..."

# 1. Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed."
    exit 1
fi

# 2. Create Virtual Environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# 3. Activate Virtual Environment
source .venv/bin/activate

# 4. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo "⬇️  Installing dependencies..."
    pip install -q -r requirements.txt
else
    echo "⚠️  Warning: requirements.txt not found."
fi

# 5. Run the Application
echo "✅ Setup complete. Launching server..."
echo "🌐 Please open http://localhost:5001 in your browser."
python -m services.app
