"""Evolution framework for MemEvolve.

Implements Pareto-based selection, mutation, and diagnosis
for memory architecture optimization.
"""

from ..utils.logging_manager import LoggingManager

from .diagnosis import (DiagnosisEngine, DiagnosisReport, FailureAnalysis,
                        FailureType, MemoryGapsAnalysis, TrajectoryStep)
from .genotype import (EncodeConfig, GenotypeFactory, ManageConfig,
                       MemoryGenotype, RetrieveConfig, StoreConfig)
from .mutation import (MutationEngine, MutationResult, MutationStrategy,
                       RandomMutationStrategy, TargetedMutationStrategy)
from .selection import EvaluationResult, FitnessMetrics, ParetoSelector

logger = LoggingManager.get_logger(__name__)

__all__ = [
    "MemoryGenotype",
    "EncodeConfig",
    "StoreConfig",
    "RetrieveConfig",
    "ManageConfig",
    "GenotypeFactory",
    "FitnessMetrics",
    "EvaluationResult",
    "ParetoSelector",
    "DiagnosisEngine",
    "DiagnosisReport",
    "TrajectoryStep",
    "MemoryGapsAnalysis",
    "FailureAnalysis",
    "FailureType",
    "MutationEngine",
    "MutationStrategy",
    "MutationResult",
    "RandomMutationStrategy",
    "TargetedMutationStrategy"
]
