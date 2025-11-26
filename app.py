"""SQL Agent Web Application Entry Point

This is the main entry point for the application.
It imports the FastAPI app from the app module.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
