from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .genotype import MemoryGenotype


@dataclass
class FitnessMetrics:
    """Fitness metrics for evaluating memory architectures."""

    performance: float = 0.0
    cost: float = 0.0
    retrieval_accuracy: float = 0.0
    storage_efficiency: float = 0.0
    response_time: float = 0.0
    memory_size_mb: float = 0.0

    def calculate_fitness_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall fitness score from metrics.

        Args:
            weights: Optional weights for each metric. If None, uses defaults.

        Returns:
            Combined fitness score (higher is better)
        """
        if weights is None:
            weights = {
                "performance": 0.4,
                "cost": -0.3,
                "retrieval_accuracy": 0.2,
                "storage_efficiency": 0.1
            }

        score = (
            weights.get("performance", 0) * self.performance +
            weights.get("cost", 0) * self.cost +
            weights.get("retrieval_accuracy", 0) * self.retrieval_accuracy +
            weights.get("storage_efficiency", 0) * self.storage_efficiency
        )

        return score


@dataclass
class EvaluationResult:
    """Result of evaluating a memory architecture."""

    genotype: MemoryGenotype
    fitness_metrics: FitnessMetrics
    fitness_score: float
    is_pareto_optimal: bool = False
    dominated_by: Optional[List[int]] = None
    dominates_list: List[int] = field(default_factory=list)

    def dominates(self, other: 'EvaluationResult') -> bool:
        """Check if this result dominates another.

        A result dominates if it's better or equal in all objectives
        and strictly better in at least one.
        """
        self_is_better = (
            self.fitness_metrics.performance >=
            other.fitness_metrics.performance and
            self.fitness_metrics.retrieval_accuracy >=
            other.fitness_metrics.retrieval_accuracy and
            self.fitness_metrics.storage_efficiency >=
            other.fitness_metrics.storage_efficiency and
            self.fitness_metrics.response_time <=
            other.fitness_metrics.response_time and
            self.fitness_metrics.memory_size_mb <=
            other.fitness_metrics.memory_size_mb
        )

        self_is_strictly_better = (
            self.fitness_metrics.performance >
            other.fitness_metrics.performance or
            self.fitness_metrics.retrieval_accuracy >
            other.fitness_metrics.retrieval_accuracy or
            self.fitness_metrics.storage_efficiency >
            other.fitness_metrics.storage_efficiency or
            self.fitness_metrics.response_time <
            other.fitness_metrics.response_time or
            self.fitness_metrics.memory_size_mb <
            other.fitness_metrics.memory_size_mb
        )

        return self_is_better and self_is_strictly_better


class ParetoSelector:
    """Selects best genotypes using Pareto ranking."""

    def __init__(
        self,
        performance_weight: float = 0.4,
        cost_weight: float = 0.3
    ):
        self.performance_weight = performance_weight
        self.cost_weight = cost_weight

    def evaluate(
        self,
        genotypes: List[MemoryGenotype],
        performance_data: Optional[Dict[str, float]] = None,
        cost_data: Optional[Dict[str, float]] = None
    ) -> List['EvaluationResult']:
        """Evaluate all genotypes and assign fitness scores.

        Args:
            genotypes: List of genotypes to evaluate
            performance_data: Optional dict mapping genome_id to performance scores
            cost_data: Optional dict mapping genome_id to cost scores

        Returns:
            List of evaluation results
        """
        if performance_data is None:
            performance_data = {}
        if cost_data is None:
            cost_data = {}

        results = []

        for genotype in genotypes:
            genome_id = genotype.get_genome_id()

            fitness_metrics = self._calculate_fitness_metrics(
                genotype,
                genome_id,
                performance_data,
                cost_data
            )

            fitness_score = fitness_metrics.calculate_fitness_score({
                "performance": self.performance_weight,
                "cost": -self.cost_weight
            })

            result = EvaluationResult(
                genotype=genotype,
                fitness_metrics=fitness_metrics,
                fitness_score=fitness_score
            )

            results.append(result)

        return results

    def _calculate_fitness_metrics(
        self,
        genotype: MemoryGenotype,
        genome_id: str,
        performance_data: Dict[str, float],
        cost_data: Dict[str, float]
    ) -> FitnessMetrics:
        """Calculate fitness metrics for a genotype.

        Performance score (higher is better):
        - Retrieved by evaluation on benchmark tasks

        Cost score (lower is better):
        - Inverse of storage size
        - Inverse of response time
        - Inverse of retrieval time

        Combining these into a single fitness score allows Pareto optimization.
        """
        performance = performance_data.get(genome_id, 0.5)

        cost = self._calculate_cost(genotype, cost_data.get(genome_id, 1.0))

        retrieval_accuracy = performance_data.get(genome_id, 0.5)

        storage_efficiency = self._calculate_storage_efficiency(genotype)

        response_time = cost_data.get(genome_id, 1.0)

        memory_size_mb = cost_data.get(genome_id, 10.0)

        return FitnessMetrics(
            performance=performance,
            cost=cost,
            retrieval_accuracy=retrieval_accuracy,
            storage_efficiency=storage_efficiency,
            response_time=response_time,
            memory_size_mb=memory_size_mb
        )

    def _calculate_cost(
        self,
        genotype: MemoryGenotype,
        base_cost: float
    ) -> float:
        """Calculate combined cost metric.

        Lower is better. Factors:
        - Storage backend complexity
        - Vector storage overhead
        - Retrieval strategy complexity
        - Management overhead
        """
        backend_cost = self._get_backend_cost(genotype.store.backend_type)
        strategy_cost = self._get_strategy_cost(
            genotype.retrieve.strategy_type)

        total_cost = base_cost * backend_cost * strategy_cost
        return total_cost

    def _get_backend_cost(self, backend_type: str) -> float:
        """Get cost multiplier for storage backend."""
        costs = {
            "json": 1.0,
            "vector": 1.5,
            "graph": 2.0
        }
        return costs.get(backend_type, 1.0)

    def _get_strategy_cost(self, strategy_type: str) -> float:
        """Get cost multiplier for retrieval strategy."""
        costs = {
            "keyword": 1.0,
            "semantic": 1.3,
            "hybrid": 1.5,
            "llm_guided": 2.0
        }
        return costs.get(strategy_type, 1.0)

    def _calculate_storage_efficiency(
        self,
        genotype: MemoryGenotype
    ) -> float:
        """Calculate storage efficiency score.

        Higher is better. Factors:
        - Persistence enabled
        - Deduplication enabled
        - Compression potential
        """
        efficiency = 0.5

        if genotype.store.enable_persistence:
            efficiency += 0.3
        if genotype.manage.deduplicate_enabled:
            efficiency += 0.1
        if genotype.store.backend_type == "vector":
            efficiency -= 0.1

        return min(max(efficiency, 0.0), 1.0)

    def select_pareto_front(
        self,
        results: List['EvaluationResult']
    ) -> List['EvaluationResult']:
        """Select Pareto front (non-dominated solutions).

        Returns:
            List of evaluation results that are not dominated
        """
        pareto_front = []

        for i, result in enumerate(results):
            is_dominated = False
            dominated_by_indices = []

            for j, other in enumerate(results):
                if i == j:
                    continue

                if other.dominates(result):
                    is_dominated = True
                    dominated_by_indices.append(j)

            if not is_dominated:
                result.dominated_by = dominated_by_indices
                result.dominates_list = []
                pareto_front.append(result)

        return pareto_front

    def select_best(
        self,
        results: List['EvaluationResult']
    ) -> EvaluationResult:
        """Select single best result by fitness score.

        Returns:
            Best evaluation result
        """
        if not results:
            raise ValueError("No results to select from")

        best = max(results, key=lambda r: r.fitness_score)

        for result in results:
            result.dominated_by = []
            result.dominates_list = []
        best.is_pareto_optimal = True

        return best

    def select_top_n(
        self,
        results: List['EvaluationResult'],
        n: int = 10
    ) -> List['EvaluationResult']:
        """Select top N results by fitness score.

        Returns:
            List of top N evaluation results
        """
        sorted_results = sorted(
            results,
            key=lambda r: r.fitness_score,
            reverse=True
        )

        for i, result in enumerate(sorted_results):
            result.dominated_by = []
            result.dominates_list = []

        return sorted_results[:n]
