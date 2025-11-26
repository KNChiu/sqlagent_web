"""Main FastAPI Application

This module initializes the FastAPI application with:
- CORS middleware configuration
- API routes registration
- Logging configuration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .logging_config import setup_logging


# Initialize logging system
setup_logging(log_level="INFO")

# Initialize FastAPI app
app = FastAPI(title="SQL Agent API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)
