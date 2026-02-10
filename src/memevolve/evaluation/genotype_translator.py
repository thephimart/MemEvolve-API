"""Genotype to MemorySystem translation utilities.

This module provides utilities to convert MemoryGenotype instances
into functional MemorySystem instances for evaluation.
"""

from ..evolution.genotype import MemoryGenotype
from ..memory_system import MemorySystem
from ..utils.logging_manager import LoggingManager
from ..utils.config import MemEvolveConfig


def create_memory_system_from_genotype(
    genotype: MemoryGenotype,
    config: MemEvolveConfig
) -> MemorySystem:
    """
    Create a MemorySystem instance from a MemoryGenotype.

    Args:
        genotype: The genotype to convert into a functional memory system
        config: Centralized configuration for API endpoints and global settings

    Returns:
        Configured MemorySystem instance

    Raises:
        RuntimeError: If memory system creation fails
    """
    logger = LoggingManager.get_logger(__name__)

    try:
        # Create memory system with full MemEvolveConfig
        real_memory_system = MemorySystem(config)

        # Create a wrapper to add experiment info
        class MemorySystemWithInfo:
            def __init__(self, memory_system, arch_name, genotype_id):
                self._memory_system = memory_system
                self.architecture_name = arch_name
                self.genotype_id = genotype_id

            def __getattr__(self, name):
                return getattr(self._memory_system, name)

        wrapped_system = MemorySystemWithInfo(
            real_memory_system,
            f"Genotype-{genotype.get_genome_id()[:8]}",
            genotype.get_genome_id()
        )

        logger.info(f"Successfully created MemorySystem for genotype {genotype.get_genome_id()}")
        return wrapped_system

    except Exception as e:
        logger.error(f"Failed to create MemorySystem from genotype: {e}")
        raise RuntimeError(f"MemorySystem creation failed: {e}")
