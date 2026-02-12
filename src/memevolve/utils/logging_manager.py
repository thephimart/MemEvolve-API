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
from typing import Optional, Union

from .config import load_config


class NoOpLogger:
    """No-operation logger that discards all messages."""

    def __init__(self, name: str):
        self.name = name

    def debug(self, msg, *args, **kwargs): pass
    def info(self, msg, *args, **kwargs): pass
    def warning(self, msg, *args, **kwargs): pass
    def error(self, msg, *args, **kwargs): pass
    def critical(self, msg, *args, **kwargs): pass
    def setLevel(self, level): pass
    def addHandler(self, handler): pass

class ExcludeLoggersFilter(logging.Filter):
    """
    Filter out log records whose logger name starts with any
    of the given prefixes.
    """

    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(self.prefixes)

class LoggingManager:
    """Centralized logging manager with directory tree mirroring."""

    _configured = False
    _log_dir: str = "./logs"
    _base_level: str = "INFO"

    @classmethod
    def _ensure_configured(cls):
        """Ensure logging system is configured once."""
        if cls._configured:
            return

        # Load configuration to get log directory
        try:
            config = load_config()
            cls._log_dir = getattr(config.logging,
                                   'log_dir',
                                   './logs') if hasattr(
                config,
                'logging') else './logs'

            # Check global logging enable flag
            if hasattr(config.logging, 'enable') and not config.logging.enable:
                cls._base_level = 'CRITICAL'  # Minimal logging when disabled
                cls._file_level = 'CRITICAL'
            else:
                cls._base_level = getattr(config.logging, 'console_level', 'INFO')
                cls._file_level = getattr(config.logging, 'file_level', 'DEBUG')
        except Exception:
            # Fallback defaults if config loading fails
            cls._log_dir = './logs'
            cls._base_level = 'INFO'
            cls._file_level = 'DEBUG'

        # Ensure log directory exists
        Path(cls._log_dir).mkdir(parents=True, exist_ok=True)

        # Configure root logger to handle external library logs with our levels
        root_logger = logging.getLogger()
        # CRITICAL FIX: Set root logger to console level to filter external DEBUG messages
        root_logger.setLevel(getattr(logging, cls._base_level.upper()))
        
        # Suppress verbose httpx logging to clean up console output
        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.DEBUG)  # Send httpx messages to DEBUG (won't show in console)
        
        # Suppress uvicorn access logging to clean up console output  
        uvicorn_logger = logging.getLogger('uvicorn.access')
        uvicorn_logger.setLevel(logging.DEBUG)  # Send uvicorn access logs to DEBUG (won't show in console)

        # Only add handlers if not already present
        if not root_logger.handlers:
            # Create console formatter (truncated for readability)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            
            # Create file formatter (full format for detailed logs)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console handler with truncated formatter
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, cls._base_level.upper(), 'INFO'))
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(
                ExcludeLoggersFilter(
                    prefixes=(
                        "httpx",
                        "httpcore",
                    )
                )
            )

            root_logger.addHandler(console_handler)

            # File handler with full formatter
            file_handler = RotatingFileHandler(
                str(Path(cls._log_dir) / 'external.log'),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, cls._file_level.upper(), 'DEBUG'))
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        cls._configured = True

    @classmethod
    def _get_log_file_path(cls, logger_name: str) -> str:
        """Convert logger name to file path mirroring source tree with exact 1:1 mapping."""

        # Handle scripts vs src/memevolve
        if logger_name.startswith('scripts.'):
            clean_name = logger_name[8:]  # Remove 'scripts.' prefix
            base_dir = Path(cls._log_dir) / 'scripts'
        elif logger_name.startswith('src.memevolve.'):
            clean_name = logger_name[15:]  # Remove 'src.memevolve.' prefix
            base_dir = Path(cls._log_dir)
        elif logger_name.startswith('memevolve.'):
            clean_name = logger_name[10:]  # Remove 'memevolve.' prefix
            base_dir = Path(cls._log_dir)
        else:
            clean_name = logger_name
            base_dir = Path(cls._log_dir)

        # Convert dots to directory separators for exact mirroring
        path_parts = clean_name.split('.')

        # Build file path - exact 1:1 mirror of source structure
        log_path = base_dir
        for part in path_parts[:-1]:  # All parts except last (module name)
            log_path = log_path / part

        # Create directories
        log_path.mkdir(parents=True, exist_ok=True)

        # Final log file - use the module name as filename (exact 1:1 mapping)
        log_file = log_path / f"{path_parts[-1]}.log"

        return str(log_file)

    @classmethod
    def get_logger(
            cls,
            name: str,
            console_level: Optional[str] = None,
            file_level: Optional[str] = None,
            create_file: bool = True
    ) -> Union[logging.Logger, NoOpLogger]:
        """Get a properly configured logger.

        Args:
            name: Logger name (usually __name__ from calling module)
            console_level: Override console level for this specific logger
            file_level: Override file level for this specific logger
            create_file: Whether to create file handler (default: True)

        Returns:
            Configured logger instance
        """
        cls._ensure_configured()

        # Check global logging enable flag
        config = load_config()
        if hasattr(config.logging, 'enable') and not config.logging.enable:
            # Return no-op logger when logging is disabled
            return NoOpLogger(name)

        if not create_file:
            return logging.getLogger(name)

        logger = logging.getLogger(name)

        # Skip if already configured
        if logger.handlers:
            return logger

        # Get console and file levels from config
        default_console_level = getattr(config.logging, 'console_level', 'INFO')
        default_file_level = getattr(config.logging, 'file_level', 'DEBUG')

        # Apply overrides if provided
        final_console_level = console_level or default_console_level
        final_file_level = file_level or default_file_level

        # Set logger to lowest level to allow all messages
        logger.setLevel(getattr(logging, min(final_console_level, final_file_level).upper()))

        # Create console formatter (truncated for readability)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Create file formatter (full format for detailed logs)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler with truncated formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, final_console_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler with full formatter (mirroring directory structure)
        if create_file:
            log_file_path = cls._get_log_file_path(name)

            # Rotating file handler (10MB max, 5 backups)
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, final_file_level.upper()))
            file_handler.setFormatter(file_formatter)
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
def get_logger(name: str, level: Optional[str] = None, create_file: bool = True):
    """Convenience function to get a logger."""
    return LoggingManager.get_logger(name, level, create_file)
