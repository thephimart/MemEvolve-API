"""
MemEvolve: Adaptive Memory System for AI Agents

A comprehensive memory management system that evolves and adapts to user interactions
through continuous learning and optimization.
"""

__version__ = "0.1.0"
__author__ = "MemEvolve Contributors"

# API imports
from .api.server import app
# Component imports
from .components import encode, manage, retrieve, store
# Core imports
from .memory_system import MemorySystem, MemorySystemConfig
from .utils.quality_scorer import ResponseQualityScorer

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
