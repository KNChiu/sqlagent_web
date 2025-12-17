"""SQL Agent Web Application Entry Point

This file serves as the entry point for running the application directly.
For production use: uvicorn app:app

Usage:
    python app.py
"""

if __name__ == "__main__":
    import uvicorn
    from app.main import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
