"""
CardioSentinel™ - Application Entrypoint
Provides seamless compatibility for `python app.py` and WSGI runners.
"""

from run import main
from backend.server import app

if __name__ == "__main__":
    main()
