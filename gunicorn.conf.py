"""
CardioSentinel | ECG Intelligence - Gunicorn Production Configuration
Configured specifically for Render Web Service deployment.
"""

import os

# Port binding: Render dynamically assigns PORT, fallback to 5000 locally
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Concurrency & Worker Architecture:
# CardioSentinel uses an in-memory patient telemetry engine with an active background
# worker thread and continuous Server-Sent Events (SSE) streaming at 25 Hz.
# 
# 1. workers = 1:
#    Ensures a single master worker process so all patient telemetry state, alarm history,
#    and clinical queues remain unified and synchronized.
#
# 2. worker_class = "gthread" and threads = 16:
#    Uses Gunicorn's built-in multithreaded worker. This allows persistent SSE connections
#    (/stream/telemetry) to stream real-time ECG waveforms while other threads concurrently
#    handle API calls (e.g. bed switches, rhythm changes, record browsing) without blocking.
#
# 3. timeout = 120:
#    Provides generous headroom for long-lived streaming connections and ML inference requests.

workers = 1
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "16"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
keepalive = 5

# Production Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
