"""
Render Startup Script
=====================
Starts the FastAPI dashboard via gunicorn + uvicorn worker.
DB init and scanning are handled by the app's lifespan hook.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

port = os.environ.get("PORT", "8000")

# Log to stdout before exec replaces this process — Render captures stdout
print(f"EdgeFinder starting on port {port}...", flush=True)
print(f"Python {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

os.execvp("gunicorn", [
    "gunicorn",
    "dashboard.app:app",
    "--workers", "1",
    "--worker-class", "uvicorn.workers.UvicornWorker",
    "--bind", f"0.0.0.0:{port}",
    "--timeout", "120",
])
