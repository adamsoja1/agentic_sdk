"""Logging configuration for the agentic framework."""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: int = logging.INFO, log_dir: str | None = None) -> None:
    """Set up logging configuration to write logs to a file.

    Args:
        log_level: The logging level (default: INFO)
        log_dir: Directory where log files will be saved.
                Defaults to 'logs' directory in the workspace.
    """
    # Create default log directory in workspace root
    if log_dir is None:
        # Get the workspace root (parent of agentic_framework package)
        workspace_root = Path(__file__).parent.parent.parent
        log_dir = workspace_root / "logs"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"agentic_framework_{timestamp}.log"

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
