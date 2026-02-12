"""
MemEvolve: Adaptive Memory System for AI Agents

A comprehensive memory management system that evolves and adapts to user interactions
through continuous learning and optimization.
"""

from .utils.quality_scorer import ResponseQualityScorer
from .memory_system import MemorySystem, MemorySystemConfig
from .components import encode, manage, retrieve, store
from .api.server import app
__version__ = "0.1.0"
__author__ = "MemEvolve Contributors"

from .utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.debug("MemEvolve package initialized")

# API imports
# Component imports
# Core imports

__all__ = [
    "MemorySystem",
    "MemorySystemConfig",
    "ResponseQualityScorer",
    "app",
    "encode",
    "retrieve",
    "store",
    "manage",
]
