"""Logging infrastructure for MemEvolve."""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 104857600,
    backup_count: int = 5
) -> logging.Logger:
    """Setup logging for MemEvolve.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Custom log format string
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    log_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    root_logger.handlers = []

    for handler in handlers:
        root_logger.addHandler(handler)

    return root_logger


# Legacy setup_component_logging removed - now handled by LoggingManager


# Legacy setup_memevolve_logging removed - now handled by LoggingManager


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_from_config(logging_config: Dict[str, Any]) -> logging.Logger:
    """Configure logging from configuration dictionary.

    Args:
        logging_config: Logging configuration dictionary

    Returns:
        Configured logger instance
    """
    return setup_logging(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("log_file"),
        log_format=logging_config.get("format"),
        max_bytes=logging_config.get("max_log_size_mb", 100) * 1024 * 1024
    )


class StructuredLogger:
    """Structured logger with consistent format."""

    def __init__(self, name: str):
        """Initialize structured logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)

    def log(self, level: str, message: str, **kwargs):
        """Log a message with structured context.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context
        """
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {context}" if context else message

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(full_message)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log("critical", message, **kwargs)
