"""
CardioSentinel™ - Vercel Serverless WSGI Handler
Routes incoming HTTP requests to Flask app for serverless execution.
"""

import sys
import os

# Add repository root to system path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Set serverless environment flags
os.environ.setdefault("CARDIOSENTINEL_DATA_DIR", "/tmp/cardiosentinel_data")
os.environ.setdefault("FLASK_ENV", "production")

from backend.server import app

# Expose WSGI application for Vercel
app.debug = False
