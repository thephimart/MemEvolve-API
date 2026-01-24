"""Evolution framework for MemEvolve.

Implements Pareto-based selection, mutation, and diagnosis
for memory architecture optimization.
"""

from .genotype import (
    MemoryGenotype,
    EncodeConfig,
    StoreConfig,
    RetrieveConfig,
    ManageConfig,
    GenotypeFactory
)
from .selection import (
    FitnessMetrics,
    EvaluationResult,
    ParetoSelector
)
from .diagnosis import (
    DiagnosisEngine,
    DiagnosisReport,
    TrajectoryStep,
    MemoryGapsAnalysis,
    FailureAnalysis,
    FailureType
)
from .mutation import (
    MutationEngine,
    MutationStrategy,
    MutationResult,
    RandomMutationStrategy,
    TargetedMutationStrategy
)

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
