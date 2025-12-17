"""
DeepAgent Middleware for SQL Agent

This module provides custom middleware components for the DeepAgent-based SQL agent.
Middleware pattern allows modular composition of agent functionality.
"""

from app.middleware.safety import SafetyMiddleware

__all__ = [
    'SafetyMiddleware',
]
