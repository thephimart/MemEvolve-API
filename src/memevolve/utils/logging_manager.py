"""
Structured logging manager for MemEvolve API.

Provides directory-tree-based logging that mirrors source code structure:
./logs/ mirrors ./src/memevolve/ for streamlined troubleshooting.

Usage:
    from memevolve.utils.logging_manager import LoggingManager
    logger = LoggingManager.get_logger(__name__)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import load_config


class LoggingManager:
    """Centralized logging manager with directory tree mirroring."""

    _configured = False
    _log_dir: Optional[str] = None
    _base_level: Optional[str] = None

    @classmethod
    def _ensure_configured(cls):
        """Ensure logging system is configured once."""
        if cls._configured:
            return

        # Load configuration to get log directory
        config = load_config()
        cls._log_dir = getattr(config.logging,
                               'log_dir',
                               './logs') if hasattr(config,
                                                    'logging') else './logs'
        cls._base_level = getattr(
            config.logging, 'level', 'INFO') if hasattr(
            config, 'logging') else 'INFO'

        # Ensure log directory exists
        Path(cls._log_dir).mkdir(parents=True, exist_ok=True)

        # Configure root logger to avoid duplicate handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, cls._base_level.upper()))

        cls._configured = True

    @classmethod
    def _get_log_file_path(cls, logger_name: str) -> str:
        """Convert logger name to file path mirroring source tree."""
        # Remove 'memevolve.' prefix if present
        if logger_name.startswith('memevolve.'):
            clean_name = logger_name[10:]  # Remove 'memevolve.' prefix (10 chars)
        else:
            clean_name = logger_name

        # Convert dots to directory separators
        path_parts = clean_name.split('.')

        # Build file path
        log_path = Path(cls._log_dir)
        for part in path_parts[:-1]:  # All parts except last (module name)
            log_path = log_path / part

        # Create directories
        log_path.mkdir(parents=True, exist_ok=True)

        # Final log file
        log_file = log_path / f"{path_parts[-1]}.log"

        return str(log_file)

    @classmethod
    def get_logger(
            cls,
            name: str,
            level: Optional[str] = None,
            create_file: bool = True) -> logging.Logger:
        """
        Get a properly configured logger.

        Args:
            name: Logger name (usually __name__ from calling module)
            level: Override level for this specific logger
            create_file: Whether to create file handler (default: True)

        Returns:
            Configured logger instance
        """
        cls._ensure_configured()

        logger = logging.getLogger(name)

        # Skip if already configured
        if logger.handlers:
            return logger

        # Set logger level
        log_level = getattr(logging, (level or cls._base_level).upper())
        logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (mirroring directory structure)
        if create_file:
            log_file_path = cls._get_log_file_path(name)

            # Rotating file handler (10MB max, 5 backups)
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

        return logger

    @classmethod
    def set_global_level(cls, level: str):
        """Set global logging level for all loggers."""
        cls._base_level = level
        logging.getLogger().setLevel(getattr(logging, level.upper()))

    @classmethod
    def get_log_directory(cls) -> str:
        """Get the current log directory."""
        cls._ensure_configured()
        return cls._log_dir


# Convenience function for easy import
def get_logger(name: str, level: Optional[str] = None, create_file: bool = True) -> logging.Logger:
    """Convenience function to get a logger."""
    return LoggingManager.get_logger(name, level, create_file)
