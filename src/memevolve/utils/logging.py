"""Logging infrastructure for MemEvolve."""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


class OperationLogger:
    """Logger for tracking memory system operations."""

    def __init__(self, enable_logging: bool = True, max_entries: int = 10000):
        """Initialize operation logger.

        Args:
            enable_logging: Whether to enable operation logging
            max_entries: Maximum number of entries to keep in memory
        """
        self.enable_logging = enable_logging
        self.max_entries = max_entries
        self.operations: list = []

    def log(self, operation: str, details: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None):
        """Log an operation.

        Args:
            operation: Operation name
            details: Operation details
            metadata: Optional metadata (e.g., agent_id, task_id)
        """
        if not self.enable_logging:
            return

        entry = {
            "operation": operation,
            "details": details,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat() + "Z"
        }

        self.operations.append(entry)

        if len(self.operations) > self.max_entries:
            self.operations = self.operations[-self.max_entries:]

    def get_operations(self,
                       operation_type: Optional[str] = None,
                       limit: Optional[int] = None) -> list:
        """Get operations, optionally filtered by type.

        Args:
            operation_type: Filter by operation type
            limit: Maximum number of entries to return

        Returns:
            List of operation entries
        """
        result = self.operations

        if operation_type:
            result = [op for op in result
                      if op["operation"] == operation_type]

        if limit:
            result = result[-limit:]

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics.

        Returns:
            Dictionary with operation statistics
        """
        if not self.operations:
            return {
                "total": 0,
                "by_type": {},
                "first_timestamp": None,
                "last_timestamp": None
            }

        by_type = {}
        for op in self.operations:
            op_type = op["operation"]
            by_type[op_type] = by_type.get(op_type, 0) + 1

        return {
            "total": len(self.operations),
            "by_type": by_type,
            "first_timestamp": self.operations[0]["timestamp"],
            "last_timestamp": self.operations[-1]["timestamp"]
        }

    def clear(self):
        """Clear all operations."""
        self.operations.clear()

    def export(self, output_path: str):
        """Export operations to file.

        Args:
            output_path: Path to export file
        """
        import json

        with open(output_path, 'w') as f:
            json.dump(self.operations, f, indent=2)


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
