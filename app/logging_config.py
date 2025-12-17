"""
Logging configuration for SQLAgent application.

This module provides centralized logging setup with:
- Console output (colored, development-friendly)
- File output (rotating logs, production-ready)
- Consistent formatting across all modules
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """
    Initialize logging configuration for the entire application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if not exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Define log format
    log_format = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatters
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        filename="logs/sqlagent.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Configure root logger (set to WARNING to suppress all third-party libraries by default)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure module-specific loggers
    module_loggers = [
        "app.routes",
        "app.agent_service",
        "app.parsers",
        "app.schema_rag",
        "app.custom_tools",
        "app.debug_utils"
    ]

    for module_name in module_loggers:
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, log_level.upper()))

    # Note: All third-party libraries inherit WARNING level from root logger
    # Only project modules (app.*) are explicitly set to INFO/DEBUG above
    logging.info("Logging system initialized successfully")
